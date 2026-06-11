from collections.abc import AsyncIterator
import json
import re
from time import perf_counter
from typing import Any

import httpx

from src.adapters.base import ChatCompletionResult, ModelBackend
from src.observability.metrics import (
    BACKEND_LATENCY,
    MODEL_BACKEND_EVAL_DURATION,
    MODEL_BACKEND_EVAL_TOKENS,
    MODEL_BACKEND_EVAL_TOKENS_PER_SECOND,
    MODEL_BACKEND_LATENCY,
    MODEL_BACKEND_MODEL_INFO,
    MODEL_BACKEND_MODEL_REQUESTS,
    MODEL_BACKEND_REQUESTS,
    OLLAMA_EVAL_DURATION,
    OLLAMA_EVAL_TOKENS,
    OLLAMA_EVAL_TOKENS_PER_SECOND,
)
from src.schemas.errors import OpenAIError
from src.schemas.models import ModelList, ModelObject
from src.services.response_history_builder import content_to_text

NANOSECONDS_PER_SECOND = 1_000_000_000


class OllamaBackend(ModelBackend):
    """Adapter for Ollama chat generation and model metadata."""

    def __init__(
        self,
        base_url: str,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.native_base_url = _native_base_url(self.base_url)
        self.timeout = timeout
        self.transport = transport
        self.headers = headers

    async def list_models(self) -> ModelList:
        data = await self._request_json("GET", "/models", operation="list_models")
        models = _model_list(data)
        for model in models.data:
            MODEL_BACKEND_MODEL_INFO.labels(backend="ollama", model=model.id).set(1)
        return models

    async def create_chat_completion(self, payload: dict[str, Any]) -> ChatCompletionResult:
        data = await self._request_json(
            "POST",
            "/api/chat",
            operation="chat_completion",
            model=_payload_model(payload),
            json_payload=_ollama_chat_payload(payload, stream=False),
        )
        _record_native_ollama_metrics(_payload_model(payload), "chat_completion", data)
        return _chat_completion_result(data, tools_requested=bool(payload.get("tools")))

    async def create_chat_completion_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        stream_payload = _ollama_chat_payload(payload, stream=True)
        model = _payload_model(payload)
        usage: dict[str, int] = {}
        finish_reason: str | None = None
        operation = "chat_completion_stream"
        status = "failed"
        started_at = perf_counter()
        try:
            with BACKEND_LATENCY.time():
                async with self._client() as client:
                    async with client.stream("POST", self._url("/api/chat"), json=stream_payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            raw = _stream_payload_from_line(line)
                            if raw is None:
                                continue
                            if raw == "[DONE]":
                                status = "completed"
                                yield {"type": "done", "usage": usage}
                                return
                            data = _stream_json(raw)
                            if data.get("usage"):
                                usage = _usage(data["usage"])
                            if _has_native_ollama_usage(data):
                                usage = _usage(_native_ollama_data(data))
                                _record_native_ollama_metrics(model, operation, data)
                            finish_reason = _stream_finish_reason(data) or finish_reason
                            reasoning_delta = _stream_reasoning_delta(data)
                            if reasoning_delta is not None:
                                yield {"type": "reasoning_delta", "delta": reasoning_delta}
                            tool_calls = _stream_tool_calls(data)
                            if tool_calls and payload.get("tools"):
                                yield {"type": "tool_calls", "tool_calls": tool_calls}
                            delta = _stream_delta(data)
                            if delta is not None:
                                yield {"type": "delta", "delta": delta}
                            if data.get("done") is True:
                                status = "completed"
                                yield _done_chunk(usage, finish_reason)
                                return
                status = "completed"
                yield _done_chunk(usage, finish_reason)
        except httpx.TimeoutException as exc:
            status = "timeout"
            raise OpenAIError("Backend stream timed out.", status_code=504, type="server_error", code="backend_timeout") from exc
        except httpx.HTTPStatusError as exc:
            status = f"http_{exc.response.status_code}"
            raise OpenAIError(f"Backend stream returned HTTP {exc.response.status_code}.", status_code=502, type="server_error", code="backend_error") from exc
        except httpx.HTTPError as exc:
            raise OpenAIError("Backend stream failed.", status_code=502, type="server_error", code="backend_error") from exc
        finally:
            self._record_backend_metrics(operation, status, started_at, model=model)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout, transport=self.transport, headers=self.headers)

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        model: str | None = None,
        json_payload: dict[str, Any] | None = None,
    ) -> Any:
        status = "failed"
        started_at = perf_counter()
        try:
            with BACKEND_LATENCY.time():
                async with self._client() as client:
                    response = await client.request(method, self._url(path), json=json_payload)
                    response.raise_for_status()
                    status = "completed"
                    return response.json()
        except httpx.TimeoutException as exc:
            status = "timeout"
            raise OpenAIError(f"{_operation_message(operation)} timed out.", status_code=504, type="server_error", code="backend_timeout") from exc
        except httpx.HTTPStatusError as exc:
            status = f"http_{exc.response.status_code}"
            raise OpenAIError(f"{_operation_message(operation)} returned HTTP {exc.response.status_code}.", status_code=502, type="server_error", code="backend_error") from exc
        except httpx.HTTPError as exc:
            raise OpenAIError(f"{_operation_message(operation)} failed.", status_code=502, type="server_error", code="backend_error") from exc
        finally:
            self._record_backend_metrics(operation, status, started_at, model=model)

    def _record_backend_metrics(self, operation: str, status: str, started_at: float, *, model: str | None = None) -> None:
        elapsed = perf_counter() - started_at
        MODEL_BACKEND_LATENCY.labels(backend="ollama", operation=operation).observe(elapsed)
        MODEL_BACKEND_REQUESTS.labels(backend="ollama", operation=operation, status=status).inc()
        MODEL_BACKEND_MODEL_REQUESTS.labels(backend="ollama", model=model or "none", operation=operation, status=status).inc()

    def _url(self, path: str) -> str:
        if path.startswith("/api/"):
            return f"{self.native_base_url}{path}"
        return f"{self.base_url}{path}"


def _chat_completion_result(data: dict[str, Any], *, tools_requested: bool) -> ChatCompletionResult:
    if "message" in data and "choices" not in data:
        return _native_chat_completion_result(data, tools_requested=tools_requested)

    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = data.get("usage", {}) or {}
    tool_calls = _normalize_tool_calls(message.get("tool_calls") or [])
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("thinking") or ""
    finish_reason = choice.get("finish_reason") or data.get("finish_reason")
    if tool_calls and not tools_requested:
        return ChatCompletionResult(content=_undeclared_tool_call_content(tool_calls, content), reasoning=reasoning, finish_reason=finish_reason, usage=_usage(usage, reasoning_text=reasoning))
    return ChatCompletionResult(content=content, reasoning=reasoning, finish_reason=finish_reason, tool_calls=tool_calls, usage=_usage(usage, reasoning_text=reasoning))


def _native_chat_completion_result(data: dict[str, Any], *, tools_requested: bool) -> ChatCompletionResult:
    message = data.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("thinking") or message.get("reasoning") or ""
    tool_calls = _normalize_tool_calls(message.get("tool_calls") or [])
    usage = _usage(_native_ollama_data(data), reasoning_text=reasoning)
    finish_reason = data.get("done_reason") or data.get("finish_reason")
    if tool_calls and not tools_requested:
        return ChatCompletionResult(content=_undeclared_tool_call_content(tool_calls, content), reasoning=reasoning, finish_reason=finish_reason, usage=usage)
    return ChatCompletionResult(content=content, reasoning=reasoning, finish_reason=finish_reason, tool_calls=tool_calls, usage=usage)


def _stream_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OpenAIError(
            "Backend stream returned invalid JSON.",
            status_code=502,
            type="server_error",
            code="backend_error",
        ) from exc


def _operation_message(operation: str) -> str:
    messages = {
        "list_models": "Backend models request",
        "chat_completion": "Backend request",
        "chat_completion_stream": "Backend stream",
    }
    return messages.get(operation, "Backend request")


def _usage(usage: dict[str, Any], *, reasoning_text: str = "") -> dict[str, Any]:
    input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", usage.get("prompt_eval_count", 0))) or 0)
    output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", usage.get("eval_count", 0))) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    output_details = usage.get("output_tokens_details") if isinstance(usage.get("output_tokens_details"), dict) else {}
    reasoning_tokens = int(output_details.get("reasoning_tokens", usage.get("reasoning_tokens", 0)) or 0)
    if reasoning_text:
        reasoning_tokens = max(reasoning_tokens, _estimate_tokens(reasoning_text))
    result: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    if reasoning_tokens:
        result["output_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return result


def _payload_model(payload: dict[str, Any]) -> str:
    return str(payload.get("model") or "unknown")


def _native_base_url(base_url: str) -> str:
    if base_url.endswith("/v1"):
        return base_url.removesuffix("/v1")
    return base_url


def _ollama_chat_payload(payload: dict[str, Any], *, stream: bool) -> dict[str, Any]:
    native: dict[str, Any] = {
        "model": _payload_model(payload),
        "messages": _ollama_messages(payload.get("messages") or []),
        "stream": stream,
    }
    if payload.get("tools"):
        native["tools"] = payload["tools"]
    if payload.get("keep_alive") is not None:
        native["keep_alive"] = payload["keep_alive"]
    think = _ollama_think(payload)
    if think is not None:
        native["think"] = think

    options = _ollama_options(payload)
    if options:
        native["options"] = options

    response_format = _ollama_response_format(payload.get("response_format"))
    if response_format is not None:
        native["format"] = response_format
    return native


def _ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    native = []
    for message in messages:
        item = dict(message)
        item = _ollama_message_content(item)
        if item.get("tool_calls"):
            item["tool_calls"] = [_ollama_tool_call(call) for call in item["tool_calls"]]
        native.append(item)
    return native


def _ollama_message_content(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, list):
        return message

    text_parts: list[str] = []
    images: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            text_parts.append(str(part))
            continue
        part_type = part.get("type")
        if part_type == "input_image":
            image_data = part.get("image_base64")
            if isinstance(image_data, str) and image_data:
                images.append(image_data)
        elif part_type == "input_file":
            text_parts.append(content_to_text([part]))
        else:
            text_parts.append(content_to_text([part]))

    mapped = dict(message)
    mapped["content"] = "\n".join(text for text in text_parts if text)
    if images:
        mapped["images"] = images
    return mapped


def _ollama_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    native = dict(call)
    function = dict(native.get("function") or {})
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            function["arguments"] = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            function["arguments"] = {"raw": arguments}
    native["function"] = function
    return native


def _ollama_options(payload: dict[str, Any]) -> dict[str, Any]:
    options = dict(payload.get("options") or {})
    if payload.get("max_tokens") is not None:
        options["num_predict"] = payload["max_tokens"]
    for key in ("temperature", "top_p", "seed", "stop"):
        if payload.get(key) is not None:
            options[key] = payload[key]
    return options


def _ollama_response_format(response_format: Any) -> Any:
    if not isinstance(response_format, dict):
        return None
    if "format" in response_format:
        return _ollama_response_format(response_format["format"])
    if response_format.get("type") == "json_object":
        return "json"
    if response_format.get("type") != "json_schema":
        return None

    json_schema = response_format.get("json_schema")
    if isinstance(json_schema, dict):
        return json_schema.get("schema", json_schema)
    return response_format.get("schema")


def _ollama_think(payload: dict[str, Any]) -> bool | str | None:
    reasoning = payload.get("reasoning")
    model = _payload_model(payload).lower()
    if not isinstance(reasoning, dict):
        if "gpt-oss" in model or model.startswith("gpt-oss"):
            return False
        return None

    effort = reasoning.get("effort")
    if effort in {"low", "medium", "high"}:
        return effort
    if effort == "xhigh":
        return "high"
    if effort in {"none", "minimal"}:
        return False
    if "gpt-oss" in model or model.startswith("gpt-oss"):
        return "medium"
    return True


def _record_native_ollama_metrics(model: str, operation: str, data: dict[str, Any]) -> None:
    native = _native_ollama_data(data)
    stages = {
        "prefill": ("prompt_eval_count", "prompt_eval_duration"),
        "decode": ("eval_count", "eval_duration"),
    }
    for stage, (count_key, duration_key) in stages.items():
        tokens = _int_metric(native.get(count_key))
        duration_seconds = _duration_seconds(native.get(duration_key))
        if tokens <= 0:
            continue

        labels = {"model": model, "operation": operation, "stage": stage}
        backend_labels = {"backend": "ollama", **labels}
        MODEL_BACKEND_EVAL_TOKENS.labels(**backend_labels).inc(tokens)
        OLLAMA_EVAL_TOKENS.labels(**labels).inc(tokens)
        if duration_seconds <= 0:
            continue
        MODEL_BACKEND_EVAL_DURATION.labels(**backend_labels).inc(duration_seconds)
        MODEL_BACKEND_EVAL_TOKENS_PER_SECOND.labels(**backend_labels).set(tokens / duration_seconds)
        OLLAMA_EVAL_DURATION.labels(**labels).inc(duration_seconds)
        OLLAMA_EVAL_TOKENS_PER_SECOND.labels(**labels).set(tokens / duration_seconds)


def _native_ollama_data(data: dict[str, Any]) -> dict[str, Any]:
    usage = data.get("usage")
    native = dict(usage) if isinstance(usage, dict) else {}
    native.update({key: value for key, value in data.items() if key != "usage"})
    return native


def _int_metric(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _duration_seconds(value: Any) -> float:
    try:
        return float(value or 0) / NANOSECONDS_PER_SECOND
    except (TypeError, ValueError):
        return 0.0


def _stream_payload_from_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("data:"):
        return stripped.removeprefix("data:").strip()
    if stripped.startswith("{"):
        return stripped
    return None


def _stream_delta(data: dict[str, Any]) -> str | None:
    choices = data.get("choices") or []
    if choices:
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        if "content" in delta:
            return delta.get("content") or ""
        message = choice.get("message") or {}
        if "content" in message:
            return message.get("content") or ""

    message = data.get("message") or {}
    if isinstance(message, dict) and "content" in message:
        return message.get("content") or ""
    if "response" in data:
        return str(data.get("response") or "")
    return None


def _stream_finish_reason(data: dict[str, Any]) -> str | None:
    choices = data.get("choices") or []
    if choices:
        choice = choices[0] or {}
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            return str(finish_reason)
    finish_reason = data.get("done_reason") or data.get("finish_reason")
    return str(finish_reason) if finish_reason else None


def _done_chunk(usage: dict[str, int], finish_reason: str | None) -> dict[str, Any]:
    chunk: dict[str, Any] = {"type": "done", "usage": usage}
    if finish_reason is not None:
        chunk["finish_reason"] = finish_reason
    return chunk


def _stream_reasoning_delta(data: dict[str, Any]) -> str | None:
    choices = data.get("choices") or []
    if choices:
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        for key in ("reasoning", "thinking"):
            if key in delta:
                return delta.get(key) or ""
        message = choice.get("message") or {}
        for key in ("reasoning", "thinking"):
            if key in message:
                return message.get(key) or ""

    message = data.get("message") or {}
    if isinstance(message, dict):
        for key in ("thinking", "reasoning"):
            if key in message:
                return message.get(key) or ""
    if "thinking" in data:
        return str(data.get("thinking") or "")
    if "reasoning" in data:
        return str(data.get("reasoning") or "")
    return None


def _stream_tool_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    choices = data.get("choices") or []
    tool_calls: Any = None
    if choices:
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        if delta.get("tool_calls"):
            tool_calls = delta.get("tool_calls")
        else:
            message = choice.get("message") or {}
            tool_calls = message.get("tool_calls")
    if tool_calls is None:
        message = data.get("message") or {}
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return _normalize_tool_calls(tool_calls)


def _has_native_ollama_usage(data: dict[str, Any]) -> bool:
    native = _native_ollama_data(data)
    return any(key in native for key in ("prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration"))


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _model_list(data: Any) -> ModelList:
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return ModelList(data=[_model_object(model) for model in data["data"]])
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return ModelList(data=[_model_object(model) for model in data["models"]])
    if isinstance(data, list):
        return ModelList(data=[_model_object(model) for model in data])
    return ModelList()


def _model_object(model: Any) -> ModelObject:
    if isinstance(model, str):
        return ModelObject(id=model)
    if not isinstance(model, dict):
        return ModelObject(id=str(model))

    model_id = model.get("id") or model.get("name") or model.get("model") or str(model)
    created = model.get("created") if isinstance(model.get("created"), int) else 0
    owned_by = model.get("owned_by") or "local"
    extra = {key: value for key, value in model.items() if key not in {"id", "name", "model", "object", "created", "owned_by"}}
    return ModelObject(id=str(model_id), created=created, owned_by=str(owned_by), **extra)


def _normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for call in tool_calls:
        function = dict(call.get("function") or {})
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, separators=(",", ":"))
        function["arguments"] = arguments
        normalized.append(
            {
                "id": call.get("id"),
                "type": call.get("type", "function"),
                "function": function,
            }
        )
    return normalized


def _undeclared_tool_call_content(tool_calls: list[dict[str, Any]], fallback: str) -> str:
    arguments = []
    for call in tool_calls:
        fn = call.get("function") or {}
        if fn.get("arguments"):
            arguments.append(str(fn["arguments"]))
    if len(arguments) == 1:
        return arguments[0]
    if arguments:
        return "[" + ",".join(arguments) + "]"
    return fallback
