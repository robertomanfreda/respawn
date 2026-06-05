import time
from collections.abc import AsyncIterator
from typing import Any

from src.adapters.base import ChatCompletionResult, ModelBackend
from src.config import Settings
from src.observability.metrics import IN_FLIGHT_RESPONSES, MODEL_TOKEN_USAGE, RESPONSE_LATENCY, RESPONSES, TOKEN_USAGE
from src.schemas.errors import OpenAIError
from src.schemas.responses import ResponseInputItemList, ResponseInputTokenCount, ResponseObject, ResponseRequest, ResponseUsage
from src.services.conversation_builder import (
    assistant_text_to_output,
    build_messages,
    response_tools_to_chat_tools,
)
from src.services.id_generator import generate_id
from src.services.prompt_cache import PromptCache, PromptCacheMatch
from src.services.structured_outputs import repair_instruction, schema_from_response_format, validate_text_against_schema
from src.services.responses_compat import (
    estimate_input_tokens,
    estimate_text_tokens,
    input_items_from_request,
    paginate_items,
    reasoning_output_item,
    reasoning_requested,
    response_output_text,
    validate_text_responses_request,
)
from src.services.tool_loop import ToolLoop
from src.services.usage_meter import enrich_response_usage, normalize_usage
from src.storage.repository import ResponseRepository
from src.streaming.events import make_event
from src.tools.registry import ToolRegistry


class ResponseService:
    """Orchestrates Responses API requests across storage, tools, and backends."""

    def __init__(self, *, settings: Settings, repository: ResponseRepository, backend: ModelBackend, prompt_cache: PromptCache, registry: ToolRegistry) -> None:
        self.settings = settings
        self.repository = repository
        self.backend = backend
        self.prompt_cache = prompt_cache
        self.registry = registry

    async def create(self, request: ResponseRequest, tenant_id: str | None) -> ResponseObject:
        validate_text_responses_request(request)
        model = request.model or self.settings.default_model
        response_id = generate_id("resp")
        should_store = self._should_store(request)
        mode = "blocking"
        status = "failed"
        started_at = time.perf_counter()
        response_created = False
        IN_FLIGHT_RESPONSES.labels(model=model, mode=mode).inc()
        try:
            if should_store:
                await self.repository.create_response(
                    response_id=response_id,
                    model=model,
                    previous_response_id=request.previous_response_id,
                    input_json=request.input,
                    request_json=request.model_dump(exclude_none=True),
                    metadata_json=request.metadata,
                    tenant_id=tenant_id,
                )
                response_created = True
            chain = await self.repository.load_chain(request.previous_response_id, tenant_id, self.settings.max_chain_depth)
            payload = self._payload(model=model, request=request, chain=chain)
            cache_match = self.prompt_cache.inspect(payload, prompt_cache_key=request.prompt_cache_key, retention=request.prompt_cache_retention)
            result = await self._run_with_structured_repair(response_id=response_id, payload=payload, request=request, should_store=should_store)
            output = self._output_items(result, request)
            usage = self._usage(result, cache_match)
            self._record_token_usage(model, usage)
            self.prompt_cache.store(payload, prompt_cache_key=request.prompt_cache_key, retention=request.prompt_cache_retention)
            if should_store:
                await self.repository.complete_response(response_id, output, usage.model_dump())
                await self.repository.session.commit()
            status = "completed"
            return self._response_object(response_id, model, "completed", output, usage, request.metadata)
        except Exception as exc:
            if should_store and response_created:
                await self.repository.fail_response(response_id, _error_json(exc))
                await self.repository.session.commit()
            if isinstance(exc, OpenAIError):
                raise
            raise OpenAIError("Internal gateway error.", status_code=500, type="server_error", code="internal_error") from exc
        finally:
            self._record_response_metrics(model=model, mode=mode, status=status, should_store=should_store, started_at=started_at)

    async def stream(self, request: ResponseRequest, tenant_id: str | None) -> AsyncIterator[str]:
        validate_text_responses_request(request)
        model = request.model or self.settings.default_model
        response_id = generate_id("resp")
        created_at = int(time.time())
        output_text = ""
        reasoning_text = ""
        usage = ResponseUsage()
        sequence_number = 0
        mode = "stream"
        status = "cancelled"
        started_at = time.perf_counter()

        def event(name: str, data: dict[str, Any]) -> str:
            nonlocal sequence_number
            rendered = make_event(name, data, sequence_number)
            sequence_number += 1
            return rendered

        should_store = self._should_store(request)
        response_created = False
        IN_FLIGHT_RESPONSES.labels(model=model, mode=mode).inc()
        try:
            if should_store:
                await self.repository.create_response(
                    response_id=response_id,
                    model=model,
                    previous_response_id=request.previous_response_id,
                    input_json=request.input,
                    request_json=request.model_dump(exclude_none=True),
                    metadata_json=request.metadata,
                    tenant_id=tenant_id,
                )
                response_created = True
            in_progress_response = self._response_object(response_id, model, "in_progress", [], usage, request.metadata, created_at=created_at).model_dump()
            yield event("response.created", {"response": in_progress_response})
            chain = await self.repository.load_chain(request.previous_response_id, tenant_id, self.settings.max_chain_depth)
            payload = self._payload(model=model, request=request, chain=chain)
            cache_match = self.prompt_cache.inspect(payload, prompt_cache_key=request.prompt_cache_key, retention=request.prompt_cache_retention)
            output_index = 0
            reasoning_id: str | None = None
            yield event("response.in_progress", {"response": in_progress_response})
            if reasoning_requested(request):
                reasoning_id = generate_id("rs")
                yield event(
                    "response.output_item.added",
                    {
                        "output_index": output_index,
                        "item": {"id": reasoning_id, "type": "reasoning", "summary": [], "status": "in_progress"},
                    },
                )
                output_index += 1
            output_id = generate_id("msg")
            yield event(
                "response.output_item.added",
                {
                    "output_index": output_index,
                    "item": {"id": output_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []},
                },
            )
            yield event(
                "response.content_part.added",
                {
                    "item_id": output_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": ""},
                },
            )
            async for chunk in self.backend.create_chat_completion_stream(payload):
                if chunk.get("type") == "reasoning_delta":
                    reasoning_text += chunk.get("delta", "")
                elif chunk.get("type") == "delta":
                    delta = chunk.get("delta", "")
                    output_text += delta
                    yield event(
                        "response.output_text.delta",
                        {
                            "item_id": output_id,
                            "output_index": output_index,
                            "content_index": 0,
                            "delta": delta,
                            "logprobs": [],
                        },
                    )
                elif chunk.get("type") == "done":
                    usage = normalize_usage(chunk.get("usage"))
            validate_text_against_schema(output_text, schema_from_response_format(self._structured_format(request)))
            usage = enrich_response_usage(
                usage,
                cached_tokens=cache_match.cached_tokens,
                reasoning_tokens=self._reasoning_tokens(reasoning_text, usage),
                minimum_output_tokens=estimate_text_tokens(output_text),
            )
            output = []
            if reasoning_id is not None:
                reasoning_item = reasoning_output_item(reasoning_id, reasoning_text, request)
                output.append(reasoning_item)
                if reasoning_item["summary"]:
                    yield event("response.reasoning_summary_text.done", {"item_id": reasoning_id, "output_index": 0, "summary_index": 0, "text": reasoning_item["summary"][0]["text"]})
                yield event("response.output_item.done", {"output_index": 0, "item": reasoning_item})
            output.append(assistant_text_to_output(output_id, output_text))
            self._record_token_usage(model, usage)
            self.prompt_cache.store(payload, prompt_cache_key=request.prompt_cache_key, retention=request.prompt_cache_retention)
            if should_store:
                await self.repository.complete_response(response_id, output, usage.model_dump())
                await self.repository.session.commit()
            yield event(
                "response.output_text.done",
                {"item_id": output_id, "output_index": output_index, "content_index": 0, "text": output_text, "logprobs": []},
            )
            yield event(
                "response.content_part.done",
                {
                    "item_id": output_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": output_text},
                },
            )
            yield event("response.output_item.done", {"output_index": output_index, "item": output[-1]})
            completed = self._response_object(response_id, model, "completed", output, usage, request.metadata, created_at=created_at).model_dump()
            status = "completed"
            yield event("response.completed", {"response": completed})
        except Exception as exc:
            status = "failed"
            if should_store and response_created:
                await self.repository.fail_response(response_id, _error_json(exc))
                await self.repository.session.commit()
            yield event("response.failed", {"response": self._response_object(response_id, model, "failed", [], usage, request.metadata, error=_error_json(exc), created_at=created_at).model_dump()})
            yield event("error", {"error": _error_json(exc)})
        finally:
            self._record_response_metrics(model=model, mode=mode, status=status, should_store=should_store, started_at=started_at)

    async def retrieve(self, response_id: str, tenant_id: str | None) -> ResponseObject:
        record = await self.repository.require_response(response_id, tenant_id)
        return self._response_object(
            record.id,
            record.model,
            record.status,
            record.output_json or [],
            normalize_usage(record.usage_json),
            record.metadata_json or {},
            error=record.error_json,
            created_at=int(record.created_at.timestamp()),
        )

    async def list_input_items(self, response_id: str, tenant_id: str | None, *, after: str | None, limit: int, order: str) -> ResponseInputItemList:
        record = await self.repository.require_response(response_id, tenant_id)
        items = input_items_from_request(record.request_json or {})
        page, has_more = paginate_items(items, after=after, limit=limit, order=order)
        return ResponseInputItemList(
            data=page,
            first_id=page[0]["id"] if page else None,
            last_id=page[-1]["id"] if page else None,
            has_more=has_more,
        )

    async def count_input_tokens(self, request: ResponseRequest, tenant_id: str | None) -> ResponseInputTokenCount:
        validate_text_responses_request(request)
        chain = await self.repository.load_chain(request.previous_response_id, tenant_id, self.settings.max_chain_depth)
        payload = self._payload(model=request.model or self.settings.default_model, request=request, chain=chain)
        cache_match = self.prompt_cache.inspect(payload, prompt_cache_key=request.prompt_cache_key, retention=request.prompt_cache_retention)
        input_tokens = cache_match.input_tokens or estimate_input_tokens(request, chain)
        return ResponseInputTokenCount(input_tokens=input_tokens, input_tokens_details={"cached_tokens": min(cache_match.cached_tokens, input_tokens)})

    async def delete(self, response_id: str, tenant_id: str | None) -> None:
        await self.repository.soft_delete(response_id, tenant_id)
        await self.repository.session.commit()

    def _payload(self, *, model: str, request: ResponseRequest, chain: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": build_messages(instructions=request.instructions, chain=chain, input_value=request.input),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_output_tokens or self.settings.max_output_tokens_default,
        }
        tools = response_tools_to_chat_tools(request.tools)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = request.tool_choice or "auto"
        structured_format = self._structured_format(request)
        if structured_format:
            payload["response_format"] = structured_format
        if request.reasoning is not None:
            payload["reasoning"] = request.reasoning
        return {k: v for k, v in payload.items() if v is not None}

    def _output_items(self, result: ChatCompletionResult, request: ResponseRequest) -> list[dict[str, Any]]:
        output = [*result.output_items]
        if reasoning_requested(request):
            output.insert(0, reasoning_output_item(generate_id("rs"), result.reasoning, request))
        if result.content or not result.unhandled_tool_calls:
            output.append(assistant_text_to_output(generate_id("msg"), result.content))
        return output

    def _usage(self, result: ChatCompletionResult, cache_match: PromptCacheMatch) -> ResponseUsage:
        usage = normalize_usage(result.usage)
        return enrich_response_usage(
            usage,
            cached_tokens=cache_match.cached_tokens,
            reasoning_tokens=self._reasoning_tokens(result.reasoning, usage),
            minimum_output_tokens=estimate_text_tokens(result.content),
        )

    def _reasoning_tokens(self, reasoning_text: str, usage: ResponseUsage) -> int:
        reported = usage.output_tokens_details.reasoning_tokens
        estimated = estimate_text_tokens(reasoning_text)
        return max(reported, estimated)

    async def _run_with_structured_repair(
        self,
        *,
        response_id: str,
        payload: dict[str, Any],
        request: ResponseRequest,
        should_store: bool,
    ):
        schema = schema_from_response_format(self._structured_format(request))
        result = await ToolLoop(
            self.backend,
            self.registry,
            self.repository if should_store else None,
            request.max_tool_calls if request.max_tool_calls is not None else self.settings.max_tool_iterations,
            self.settings.tool_timeout_seconds,
        ).run(response_id=response_id, payload=payload)
        try:
            validate_text_against_schema(result.content, schema)
            return result
        except OpenAIError as exc:
            if not schema or exc.code != "structured_output_validation_failed":
                raise

        repair_payload = dict(payload)
        repair_payload["messages"] = [
            *payload["messages"],
            {"role": "assistant", "content": result.content},
            {"role": "system", "content": repair_instruction(schema)},
        ]
        repaired = await ToolLoop(
            self.backend,
            self.registry,
            self.repository if should_store else None,
            request.max_tool_calls if request.max_tool_calls is not None else self.settings.max_tool_iterations,
            self.settings.tool_timeout_seconds,
        ).run(response_id=response_id, payload=repair_payload)
        validate_text_against_schema(repaired.content, schema)
        return repaired

    def _should_store(self, request: ResponseRequest) -> bool:
        return self.settings.store_default if request.store is None else request.store

    def _structured_format(self, request: ResponseRequest) -> dict[str, Any] | None:
        if request.response_format:
            return request.response_format
        return request.text

    def _record_token_usage(self, model: str, usage: ResponseUsage) -> None:
        TOKEN_USAGE.labels(model=model, kind="input").inc(usage.input_tokens)
        TOKEN_USAGE.labels(model=model, kind="output").inc(usage.output_tokens)
        TOKEN_USAGE.labels(model=model, kind="total").inc(usage.total_tokens)
        TOKEN_USAGE.labels(model=model, kind="cached_input").inc(usage.input_tokens_details.cached_tokens)
        TOKEN_USAGE.labels(model=model, kind="reasoning").inc(usage.output_tokens_details.reasoning_tokens)
        MODEL_TOKEN_USAGE.labels(api="responses", model=model, kind="input").inc(usage.input_tokens)
        MODEL_TOKEN_USAGE.labels(api="responses", model=model, kind="output").inc(usage.output_tokens)
        MODEL_TOKEN_USAGE.labels(api="responses", model=model, kind="total").inc(usage.total_tokens)
        MODEL_TOKEN_USAGE.labels(api="responses", model=model, kind="cached_input").inc(usage.input_tokens_details.cached_tokens)
        MODEL_TOKEN_USAGE.labels(api="responses", model=model, kind="reasoning").inc(usage.output_tokens_details.reasoning_tokens)

    def _record_response_metrics(self, *, model: str, mode: str, status: str, should_store: bool, started_at: float) -> None:
        RESPONSE_LATENCY.labels(model=model, mode=mode).observe(time.perf_counter() - started_at)
        RESPONSES.labels(model=model, mode=mode, status=status, stored=str(should_store).lower()).inc()
        IN_FLIGHT_RESPONSES.labels(model=model, mode=mode).dec()

    def _response_object(
        self,
        response_id: str,
        model: str,
        status: str,
        output: list[dict[str, Any]],
        usage: ResponseUsage,
        metadata: dict[str, Any],
        *,
        error: dict[str, Any] | None = None,
        created_at: int | None = None,
    ) -> ResponseObject:
        return ResponseObject(
            id=response_id,
            created_at=created_at or int(time.time()),
            status=status,
            error=error,
            model=model,
            output=output,
            output_text=response_output_text(output),
            usage=usage,
            metadata=metadata,
        )


def _error_json(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, OpenAIError):
        return exc.to_response()["error"]
    return {"message": str(exc), "type": "server_error", "param": None, "code": "internal_error"}
