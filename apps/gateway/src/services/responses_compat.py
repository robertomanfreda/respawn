import re
from typing import Any

from src.schemas.errors import OpenAIError
from src.schemas.responses import ResponseRequest
from src.services.conversation_builder import content_to_text


UNSUPPORTED_FIELDS = {
    "conversation": "Conversation objects are not supported yet. Use previous_response_id for stored local state.",
    "context_management": "Context management is not supported yet.",
    "include": "Include expansions are not supported yet.",
    "parallel_tool_calls": "Parallel tool calls are not supported yet.",
    "prompt": "Hosted prompt templates are not supported yet.",
    "service_tier": "Service tiers are not supported by the local gateway.",
    "top_logprobs": "Top logprobs are not supported yet.",
    "user": "The deprecated user field is not supported by Respawn.",
}

TEXT_CONTENT_TYPES = {"input_text", "output_text", "text"}
SUPPORTED_TOOL_TYPES = {"function"}
SUPPORTED_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high"}
SUPPORTED_REASONING_SUMMARIES = {"auto", "concise", "detailed"}
SUPPORTED_PROMPT_CACHE_RETENTIONS = {"in_memory", "24h"}


def validate_text_responses_request(request: ResponseRequest) -> None:
    if request.background:
        _unsupported("background", "Background mode is not supported yet.")
    if request.truncation == "auto":
        _unsupported("truncation", "Automatic truncation is not supported yet.")

    for field, message in UNSUPPORTED_FIELDS.items():
        value = getattr(request, field)
        if value not in (None, [], {}):
            _unsupported(field, message)

    if request.tool_choice not in (None, "auto"):
        _unsupported("tool_choice", "Only tool_choice='auto' is supported today.")

    _validate_reasoning(request.reasoning)
    _validate_prompt_cache(request.prompt_cache_key, request.prompt_cache_retention)

    for index, tool in enumerate(request.tools):
        tool_type = tool.get("type")
        if tool_type not in SUPPORTED_TOOL_TYPES:
            _unsupported(f"tools.{index}.type", f"Tool type '{tool_type}' is not supported yet.")

    validate_text_input(request.input)
    _validate_text_format(request.text or request.response_format)


def validate_text_input(input_value: str | list[dict[str, Any]] | None, *, param: str = "input") -> None:
    if input_value is None or isinstance(input_value, str):
        return
    if not isinstance(input_value, list):
        raise OpenAIError("input must be a string or a list of input items.", param=param)

    for index, item in enumerate(input_value):
        if not isinstance(item, dict):
            raise OpenAIError("input items must be objects.", param=f"{param}.{index}")

        item_type = item.get("type")
        role = item.get("role")
        if item_type == "message" or role in {"user", "assistant", "system", "developer"}:
            _validate_text_content(item.get("content", ""), param=f"{param}.{index}.content")
            continue
        if item_type == "reasoning":
            continue
        if item_type in {"function_call", "function_call_output", "tool_result"}:
            continue
        if item_type in {"input_image", "input_file", "input_audio"}:
            _unsupported(f"{param}.{index}.type", f"Input item type '{item_type}' is not supported in text-only compatibility.")
        _unsupported(f"{param}.{index}.type", f"Input item type '{item_type}' is not supported.")


def input_items_from_request(request_json: dict[str, Any]) -> list[dict[str, Any]]:
    input_value = request_json.get("input")
    if isinstance(input_value, str):
        return [_message_item("msg_input_0", "user", input_value)]
    if not isinstance(input_value, list):
        return []

    items = []
    for index, item in enumerate(input_value):
        item_id = str(item.get("id") or item.get("call_id") or f"input_{index}")
        item_type = item.get("type")
        role = item.get("role")
        if item_type == "message" or role in {"user", "assistant", "system", "developer"}:
            items.append(_message_item(item_id, role or "user", item.get("content", "")))
        elif item_type == "function_call":
            items.append(
                {
                    "id": item_id,
                    "type": "function_call",
                    "call_id": item.get("call_id") or item_id,
                    "name": item.get("name"),
                    "arguments": item.get("arguments", "{}"),
                    "status": item.get("status", "completed"),
                }
            )
        elif item_type in {"function_call_output", "tool_result"}:
            items.append(
                {
                    "id": item_id,
                    "type": "function_call_output",
                    "call_id": item.get("call_id") or item.get("tool_call_id"),
                    "output": item.get("output", item.get("content", "")),
                    "status": item.get("status", "completed"),
                }
            )
        elif item_type == "reasoning":
            items.append(
                {
                    "id": item_id,
                    "type": "reasoning",
                    "summary": item.get("summary", []),
                    "status": item.get("status", "completed"),
                }
            )
    return items


def paginate_items(items: list[dict[str, Any]], *, after: str | None, limit: int, order: str) -> tuple[list[dict[str, Any]], bool]:
    if order not in {"asc", "desc"}:
        raise OpenAIError("order must be 'asc' or 'desc'.", param="order", code="invalid_order")
    if limit < 1 or limit > 100:
        raise OpenAIError("limit must be between 1 and 100.", param="limit", code="invalid_limit")

    ordered = items if order == "asc" else list(reversed(items))
    if after:
        try:
            start = next(index + 1 for index, item in enumerate(ordered) if item.get("id") == after)
        except StopIteration as exc:
            raise OpenAIError("Input item cursor not found.", status_code=404, param="after", code="not_found") from exc
        ordered = ordered[start:]

    page = ordered[:limit]
    return page, len(ordered) > limit


def estimate_input_tokens(request: ResponseRequest, chain: list[dict[str, Any]] | None = None) -> int:
    parts = []
    if request.instructions:
        parts.append(request.instructions)
    for response in chain or []:
        request_json = response.get("request_json") or {}
        parts.append(_input_to_text(request_json.get("input")))
        parts.append(_output_to_text(response.get("output_json") or []))
    parts.append(_input_to_text(request.input))
    if request.tools:
        parts.append(str(request.tools))
    return _estimate_tokens("\n".join(part for part in parts if part))


def response_output_text(output: list[dict[str, Any]]) -> str:
    return _output_to_text(output)


def reasoning_requested(request: ResponseRequest) -> bool:
    return isinstance(request.reasoning, dict)


def reasoning_summary_requested(request: ResponseRequest) -> bool:
    reasoning = request.reasoning or {}
    return isinstance(reasoning, dict) and reasoning.get("summary") in SUPPORTED_REASONING_SUMMARIES


def reasoning_output_item(item_id: str, reasoning_text: str, request: ResponseRequest) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": item_id,
        "type": "reasoning",
        "summary": [],
    }
    if reasoning_summary_requested(request):
        item["summary"] = [{"type": "summary_text", "text": reasoning_summary(reasoning_text)}]
    return item


def reasoning_summary(reasoning_text: str) -> str:
    tokens = estimate_text_tokens(reasoning_text)
    if tokens <= 0:
        return "No reasoning trace was returned by the local backend."
    return f"Local backend returned a reasoning trace before the final answer. Estimated reasoning tokens: {tokens}. Raw reasoning content is intentionally not exposed by Respawn."


def estimate_text_tokens(text: str) -> int:
    return _estimate_tokens(text)


def _message_item(item_id: str, role: str, content: Any) -> dict[str, Any]:
    mapped_role = "system" if role == "developer" else role
    return {
        "id": item_id,
        "type": "message",
        "role": mapped_role,
        "content": _content_parts(content),
    }


def _content_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text", part.get("output_text", ""))
            else:
                text = str(part)
            parts.append({"type": "input_text", "text": str(text)})
        return parts
    return [{"type": "input_text", "text": content_to_text(content)}]


def _validate_text_content(content: Any, *, param: str) -> None:
    if isinstance(content, str):
        return
    if isinstance(content, list):
        for index, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in TEXT_CONTENT_TYPES:
                continue
            if part_type in {"input_image", "input_file", "input_audio"}:
                _unsupported(f"{param}.{index}.type", f"Content part type '{part_type}' is not supported in text-only compatibility.")
            _unsupported(f"{param}.{index}.type", f"Content part type '{part_type}' is not supported.")
        return
    if isinstance(content, dict):
        part_type = content.get("type")
        if part_type in TEXT_CONTENT_TYPES:
            return
        _unsupported(f"{param}.type", f"Content part type '{part_type}' is not supported.")


def _validate_text_format(response_format: dict[str, Any] | None) -> None:
    if not response_format:
        return
    if "format" in response_format:
        _validate_text_format(response_format.get("format"))
        return
    format_type = response_format.get("type")
    if format_type in {None, "text", "json_object", "json_schema"}:
        return
    _unsupported("text.format.type", f"Text format '{format_type}' is not supported.")


def _validate_reasoning(reasoning: dict[str, Any] | None) -> None:
    if reasoning is None:
        return
    if not isinstance(reasoning, dict):
        raise OpenAIError("reasoning must be an object.", param="reasoning")

    effort = reasoning.get("effort")
    if effort is not None and effort not in SUPPORTED_REASONING_EFFORTS:
        _unsupported("reasoning.effort", f"Reasoning effort '{effort}' is not supported by the local gateway.")

    summary = reasoning.get("summary")
    if summary is not None and summary not in SUPPORTED_REASONING_SUMMARIES:
        _unsupported("reasoning.summary", f"Reasoning summary '{summary}' is not supported.")

    unsupported_keys = sorted(set(reasoning) - {"effort", "summary"})
    if unsupported_keys:
        _unsupported(f"reasoning.{unsupported_keys[0]}", f"Reasoning field '{unsupported_keys[0]}' is not supported.")


def _validate_prompt_cache(prompt_cache_key: str | None, prompt_cache_retention: str | None) -> None:
    if prompt_cache_key is not None and not prompt_cache_key.strip():
        raise OpenAIError("prompt_cache_key must not be empty.", param="prompt_cache_key")
    if prompt_cache_retention is not None and prompt_cache_retention not in SUPPORTED_PROMPT_CACHE_RETENTIONS:
        _unsupported("prompt_cache_retention", f"Prompt cache retention '{prompt_cache_retention}' is not supported.")


def _input_to_text(input_value: Any) -> str:
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, list):
        return "\n".join(content_to_text(item.get("content", item.get("output", ""))) for item in input_value if isinstance(item, dict))
    return ""


def _output_to_text(output: list[dict[str, Any]]) -> str:
    parts = []
    for item in output:
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                parts.append(str(content.get("text", "")))
    return "".join(parts)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _unsupported(param: str, message: str) -> None:
    raise OpenAIError(message, status_code=400, param=param, code="unsupported_parameter")
