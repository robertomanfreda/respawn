import json
import re
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from src.config import Settings
from src.observability.metrics import FUNCTION_TOOL_UNSUPPORTED
from src.schemas.errors import OpenAIError
from src.schemas.responses import ResponseRequest
from src.services.context_management import validate_compaction_item, validate_context_management
from src.services.include_expansions import REASONING_ENCRYPTED_CONTENT, validate_include_values
from src.services.model_capabilities import reasoning_efforts_for_model
from src.services.response_history_builder import content_to_text
from src.services.reasoning_summaries import DeterministicReasoningSummaryProvider, ReasoningSummaryProvider, estimate_text_tokens


UNSUPPORTED_FIELDS = {
    "user": "The deprecated user field is not supported by Respawn.",
}

TEXT_CONTENT_TYPES = {"input_text", "output_text", "text"}
MULTIMODAL_CONTENT_TYPES = {"input_image", "input_file"}
FUNCTION_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DOMAIN_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
SUPPORTED_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
SUPPORTED_REASONING_SUMMARIES = {"auto", "concise", "detailed"}
SUPPORTED_PROMPT_CACHE_RETENTIONS = {"in_memory", "24h"}
SUPPORTED_SERVICE_TIERS = {"auto", "default", "flex", "scale", "priority"}
SUPPORTED_TOOL_CHOICE_STRINGS = {"auto", "none", "required"}
WEB_SEARCH_TOOL_TYPES = {"web_search", "web_search_preview"}
WEB_SEARCH_INTERNAL_TOOL_NAME = "respawn_web_search"
WEB_SEARCH_CONTEXT_SIZES = {"low", "medium", "high"}
WEB_SEARCH_TOOL_FIELDS = {"type", "search_context_size", "filters", "user_location", "external_web_access"}
WEB_SEARCH_FILTER_FIELDS = {"allowed_domains", "blocked_domains"}
WEB_SEARCH_MAX_DOMAINS = 100
IMAGE_GENERATION_TOOL_TYPE = "image_generation"
IMAGE_GENERATION_INTERNAL_TOOL_NAME = "respawn_image_generation"
IMAGE_GENERATION_TOOL_FIELDS = {"type", "size", "quality", "output_format", "background", "partial_images", "action", "moderation"}
IMAGE_GENERATION_QUALITIES = {"auto", "low", "medium", "high"}
IMAGE_GENERATION_OUTPUT_FORMATS = {"png"}
IMAGE_GENERATION_BACKGROUNDS = {"auto", "opaque"}
IMAGE_GENERATION_ACTIONS = {"auto", "generate"}
PROTOCOL_INPUT_ITEM_TYPES = {"web_search_call", "image_generation_call"}
IMAGE_GENERATION_SIZE_RE = re.compile(r"^([1-9][0-9]{1,4})x([1-9][0-9]{1,4})$")
METADATA_MAX_PAIRS = 16
METADATA_KEY_MAX_LENGTH = 64
METADATA_VALUE_MAX_LENGTH = 512


def validate_text_responses_request(request: ResponseRequest) -> None:
    if request.background and request.stream:
        _unsupported("stream", "Streaming background responses are not supported yet.")
    if request.background and request.store is False:
        raise OpenAIError("Background mode requires store=true.", param="store", code="invalid_request")
    _validate_stream_options(request)
    validate_context_management(request.context_management)

    validate_include_values(request.include)
    for field, message in UNSUPPORTED_FIELDS.items():
        value = getattr(request, field)
        if value not in (None, [], {}):
            _unsupported(field, message)

    _validate_function_tool_protocol(request)
    _validate_metadata(request.metadata)
    _validate_service_tier(request.service_tier)
    _validate_reasoning(request.reasoning)
    _validate_prompt_cache(request.prompt_cache_key, request.prompt_cache_retention)

    validate_text_input(request.input)
    _validate_text_format(normalized_text_config(request))


def validate_reasoning_capabilities(request: ResponseRequest, *, model: str, settings: Settings) -> None:
    if request.reasoning is None:
        return
    efforts = reasoning_efforts_for_model(model, settings)
    if not efforts:
        raise OpenAIError(
            f"Model '{model}' is not configured with the reasoning capability.",
            status_code=400,
            param="model",
            code="unsupported_model_capability",
        )

    effort = request.reasoning.get("effort") if isinstance(request.reasoning, dict) else None
    if effort is not None and effort not in efforts:
        raise OpenAIError(
            f"Reasoning effort '{effort}' is not supported by model '{model}'.",
            status_code=400,
            param="reasoning.effort",
            code="unsupported_model_capability",
        )


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
            _validate_reasoning_item(item, param=f"{param}.{index}")
            continue
        if item_type == "compaction":
            validate_compaction_item(item, param=f"{param}.{index}")
            continue
        if item_type == "function_call":
            _validate_function_call_item(item, param=f"{param}.{index}")
            continue
        if item_type == "function_call_output":
            _validate_function_call_output_item(item, param=f"{param}.{index}")
            continue
        if item_type in PROTOCOL_INPUT_ITEM_TYPES:
            _validate_protocol_input_item(item, param=f"{param}.{index}")
            continue
        if item_type == "tool_result":
            _unsupported(f"{param}.{index}.type", "Legacy tool_result input items are not supported by Respawn.")
        if item_type in {"input_image", "input_file"}:
            continue
        if item_type == "input_audio":
            _unsupported(f"{param}.{index}.type", "Audio input is not supported by Respawn.")
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
            items.append(_function_call_item(item, fallback_id=f"input_{index}"))
        elif item_type == "function_call_output":
            items.append(_function_call_output_item(item, fallback_id=f"input_{index}"))
        elif item_type == "web_search_call":
            items.append(_web_search_call_item(item, fallback_id=item_id))
        elif item_type == "image_generation_call":
            items.append(_image_generation_call_item(item, fallback_id=item_id))
        elif item_type == "tool_result":
            _unsupported(f"input.{index}.type", "Legacy tool_result input items are not supported by Respawn.")
        elif item_type == "reasoning":
            items.append(_reasoning_item(item, fallback_id=item_id))
        elif item_type == "compaction":
            items.append(_compaction_item(item, fallback_id=item_id))
    return items


def stream_obfuscation_enabled(request: ResponseRequest) -> bool:
    stream_options = request.stream_options or {}
    return bool(stream_options.get("include_obfuscation", True))


def paginate_items(items: list[dict[str, Any]], *, after: str | None, before: str | None = None, limit: int, order: str) -> tuple[list[dict[str, Any]], bool]:
    if order not in {"asc", "desc"}:
        raise OpenAIError("order must be 'asc' or 'desc'.", param="order", code="invalid_order")
    if limit < 1 or limit > 100:
        raise OpenAIError("limit must be between 1 and 100.", param="limit", code="invalid_limit")
    if after and before:
        raise OpenAIError("after and before cannot be used together.", param="before", code="invalid_cursor")

    ordered = items if order == "asc" else list(reversed(items))
    if after:
        try:
            start = next(index + 1 for index, item in enumerate(ordered) if item.get("id") == after)
        except StopIteration as exc:
            raise OpenAIError("Input item cursor not found.", status_code=404, param="after", code="not_found") from exc
        ordered = ordered[start:]
    if before:
        try:
            end = next(index for index, item in enumerate(ordered) if item.get("id") == before)
        except StopIteration as exc:
            raise OpenAIError("Input item cursor not found.", status_code=404, param="before", code="not_found") from exc
        ordered = ordered[:end]

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


def backend_function_tools(request: ResponseRequest) -> list[dict[str, Any]]:
    tools = _normalized_function_tools(request.tools)
    tool_choice = request.tool_choice
    backend_tools: list[dict[str, Any]]
    if isinstance(tool_choice, str) or tool_choice is None:
        backend_tools = [_backend_function_tool(tool) for tool in tools]
    elif not isinstance(tool_choice, dict) or tool_choice.get("type") != "allowed_tools":
        backend_tools = [_backend_function_tool(tool) for tool in tools]
    else:
        allowed_names = {_tool_choice_backend_function_name(tool) for tool in tool_choice.get("tools") or [] if isinstance(tool, dict)}
        backend_tools = [_backend_function_tool(tool) for tool in tools if tool["function"]["name"] in allowed_names]

    if _include_internal_web_search_tool(request):
        backend_tools.append(_backend_web_search_tool())
    if _include_internal_image_generation_tool(request):
        backend_tools.append(_backend_image_generation_tool())
    return backend_tools


def backend_tool_choice(request: ResponseRequest) -> str | dict[str, Any] | None:
    tool_choice = request.tool_choice
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return None
    choice_type = tool_choice.get("type")
    if choice_type == "function":
        name = _tool_choice_backend_function_name(tool_choice)
        return {"type": "function", "function": {"name": name}}
    if choice_type == "allowed_tools":
        return tool_choice.get("mode", "auto")
    if choice_type == IMAGE_GENERATION_TOOL_TYPE:
        return None
    return None


def web_search_tools(request: ResponseRequest) -> list[dict[str, Any]]:
    tools = []
    for index, tool in enumerate(request.tools):
        if isinstance(tool, dict) and tool.get("type") in WEB_SEARCH_TOOL_TYPES:
            tools.append(_normalized_web_search_tool(tool, param=f"tools.{index}"))
    return tools


def web_search_requested(request: ResponseRequest) -> bool:
    return bool(web_search_tools(request))


def web_search_disabled_by_choice(request: ResponseRequest) -> bool:
    return request.tool_choice == "none"


def web_search_required(request: ResponseRequest) -> bool:
    if not web_search_requested(request) or web_search_disabled_by_choice(request):
        return False
    tool_choice = request.tool_choice
    return tool_choice == "required" or (isinstance(tool_choice, dict) and tool_choice.get("type") == "web_search")


def image_generation_tools(request: ResponseRequest) -> list[dict[str, Any]]:
    tools = []
    for index, tool in enumerate(request.tools):
        if isinstance(tool, dict) and tool.get("type") == IMAGE_GENERATION_TOOL_TYPE:
            tools.append(_normalized_image_generation_tool(tool, param=f"tools.{index}"))
    return tools


def image_generation_requested(request: ResponseRequest) -> bool:
    return bool(image_generation_tools(request))


def image_generation_disabled_by_choice(request: ResponseRequest) -> bool:
    return request.tool_choice == "none"


def image_generation_required(request: ResponseRequest) -> bool:
    if not image_generation_requested(request) or image_generation_disabled_by_choice(request):
        return False
    tool_choice = request.tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == IMAGE_GENERATION_TOOL_TYPE:
        return True
    if tool_choice != "required":
        return False
    return not _normalized_function_tools(request.tools) and not web_search_requested(request)


def is_internal_image_generation_tool_call(call: dict[str, Any]) -> bool:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    return function.get("name") == IMAGE_GENERATION_INTERNAL_TOOL_NAME


def is_internal_web_search_tool_call(call: dict[str, Any]) -> bool:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    return function.get("name") == WEB_SEARCH_INTERNAL_TOOL_NAME


def tool_choice_instruction(request: ResponseRequest) -> str | None:
    if not _normalized_function_tools(request.tools):
        return None
    tool_choice = request.tool_choice
    if tool_choice == "required":
        if web_search_requested(request):
            return None
        return "You must call one of the provided functions instead of answering in natural language."
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return f"You must call the function named {_tool_choice_backend_function_name(tool_choice)} instead of answering in natural language."
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "allowed_tools" and tool_choice.get("mode") == "required":
        names = ", ".join(_tool_choice_backend_function_name(tool) for tool in tool_choice.get("tools") or [] if isinstance(tool, dict))
        return f"You must call one of these functions instead of answering in natural language: {names}."
    if request.parallel_tool_calls is False:
        return "If you call a function, call at most one function in this turn."
    return None


def function_call_output_item(item_id: str, *, call_id: str, output: Any, status: str = "completed") -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
        "status": status,
    }


def function_call_output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, separators=(",", ":"), ensure_ascii=False)


def function_call_output_name(function_name: str, request: ResponseRequest) -> dict[str, str]:
    lookup = _function_tool_call_lookup(request)
    candidate = lookup.get(function_name)
    if candidate is None:
        candidate = lookup.get(_canonical_flat_tool_name(function_name))
    if candidate is not None:
        return dict(candidate)

    by_original: dict[str, list[dict[str, str]]] = {}
    for entry in lookup.values():
        by_original.setdefault(entry["name"], []).append(entry)
    matches = by_original.get(function_name, [])
    if len(matches) == 1:
        return dict(matches[0])
    return {"name": function_name}


def function_call_output_ids(input_value: Any) -> set[str]:
    if not isinstance(input_value, list):
        return set()
    return {
        str(item.get("call_id"))
        for item in input_value
        if isinstance(item, dict) and item.get("type") == "function_call_output" and item.get("call_id")
    }


def available_function_call_ids(chain: list[dict[str, Any]], input_value: Any) -> set[str]:
    call_ids: set[str] = set()
    for response in chain or []:
        for item in response.get("output_json") or []:
            if isinstance(item, dict) and item.get("type") == "function_call" and item.get("call_id"):
                call_ids.add(str(item["call_id"]))
    if isinstance(input_value, list):
        for item in input_value:
            if isinstance(item, dict) and item.get("type") == "function_call" and item.get("call_id"):
                call_ids.add(str(item["call_id"]))
    return call_ids


def validate_function_call_outputs_match(chain: list[dict[str, Any]], input_value: Any) -> None:
    outputs = function_call_output_ids(input_value)
    if not outputs:
        return
    available = available_function_call_ids(chain, input_value)
    missing = sorted(outputs - available)
    if missing:
        raise OpenAIError(
            "function_call_output call_id does not match a function_call in the current input or previous response chain.",
            param="input",
            code="invalid_tool_call_output",
        )


def response_output_text(output: list[dict[str, Any]]) -> str:
    return _output_to_text(output)


def normalized_text_config(request: ResponseRequest | dict[str, Any] | None) -> dict[str, Any]:
    request_data = request.model_dump(exclude_none=True) if isinstance(request, ResponseRequest) else request or {}
    text = request_data.get("text")
    if isinstance(text, dict) and text:
        if "format" in text:
            return text
        return {"format": text}
    response_format = request_data.get("response_format")
    if isinstance(response_format, dict) and response_format:
        return {"format": response_format}
    return {"format": {"type": "text"}}


def reasoning_requested(request: ResponseRequest) -> bool:
    return isinstance(request.reasoning, dict)


def reasoning_summary_requested(request: ResponseRequest) -> bool:
    reasoning = request.reasoning or {}
    return isinstance(reasoning, dict) and reasoning.get("summary") in SUPPORTED_REASONING_SUMMARIES


def reasoning_encrypted_content_requested(request: ResponseRequest) -> bool:
    return REASONING_ENCRYPTED_CONTENT in request.include


def reasoning_output_item(
    item_id: str,
    reasoning_text: str,
    request: ResponseRequest,
    *,
    encrypted_content: str | None = None,
    summary_provider: ReasoningSummaryProvider | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": item_id,
        "type": "reasoning",
        "summary": [],
        "status": "completed",
    }
    if reasoning_summary_requested(request):
        provider = summary_provider or DeterministicReasoningSummaryProvider()
        item["summary"] = [{"type": "summary_text", "text": provider.summarize(reasoning_text, mode=(request.reasoning or {}).get("summary"))}]
    if encrypted_content is not None:
        item["encrypted_content"] = encrypted_content
    return item


def reasoning_summary(reasoning_text: str) -> str:
    return DeterministicReasoningSummaryProvider().summarize(reasoning_text)


def _reasoning_item(item: dict[str, Any], *, fallback_id: str) -> dict[str, Any]:
    normalized = {
        "id": str(item.get("id") or fallback_id),
        "type": "reasoning",
        "summary": item.get("summary", []),
        "status": item.get("status", "completed"),
    }
    if item.get("encrypted_content") is not None:
        normalized["encrypted_content"] = item.get("encrypted_content")
    return normalized


def _compaction_item(item: dict[str, Any], *, fallback_id: str) -> dict[str, Any]:
    normalized = {
        "id": str(item.get("id") or fallback_id),
        "type": "compaction",
        "encrypted_content": str(item.get("encrypted_content") or ""),
        "status": item.get("status", "completed"),
    }
    return normalized


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
                part_type = part.get("type")
                if part_type in MULTIMODAL_CONTENT_TYPES:
                    parts.append(dict(part))
                    continue
                text = part.get("text", part.get("output_text", ""))
            else:
                text = str(part)
            parts.append({"type": "input_text", "text": str(text)})
        return parts
    if isinstance(content, dict) and content.get("type") in MULTIMODAL_CONTENT_TYPES:
        return [dict(content)]
    return [{"type": "input_text", "text": content_to_text(content)}]


def _function_call_item(item: dict[str, Any], *, fallback_id: str) -> dict[str, Any]:
    normalized = {
        "id": str(item.get("id") or fallback_id),
        "type": "function_call",
        "call_id": str(item["call_id"]),
        "name": str(item["name"]),
        "arguments": _arguments_to_string(item.get("arguments", "{}")),
        "status": item.get("status", "completed"),
    }
    if item.get("namespace") is not None:
        normalized["namespace"] = str(item["namespace"])
    return normalized


def _function_call_output_item(item: dict[str, Any], *, fallback_id: str) -> dict[str, Any]:
    return function_call_output_item(
        str(item.get("id") or fallback_id),
        call_id=str(item["call_id"]),
        output=item.get("output", ""),
        status=item.get("status", "completed"),
    )


def _web_search_call_item(item: dict[str, Any], *, fallback_id: str) -> dict[str, Any]:
    action = item.get("action")
    return {
        "id": str(item.get("id") or fallback_id),
        "type": "web_search_call",
        "status": item.get("status", "completed"),
        "action": dict(action) if isinstance(action, dict) else {},
    }


def _image_generation_call_item(item: dict[str, Any], *, fallback_id: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "id": str(item.get("id") or fallback_id),
        "type": "image_generation_call",
        "status": item.get("status", "completed"),
        "result": str(item.get("result") or ""),
        "revised_prompt": str(item.get("revised_prompt") or ""),
        "size": str(item.get("size") or ""),
        "quality": str(item.get("quality") or ""),
        "output_format": str(item.get("output_format") or "png"),
    }
    if item.get("seed") is not None:
        normalized["seed"] = item.get("seed")
    return normalized


def _validate_text_content(content: Any, *, param: str) -> None:
    if isinstance(content, str):
        return
    if isinstance(content, list):
        for index, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in TEXT_CONTENT_TYPES or part_type in MULTIMODAL_CONTENT_TYPES:
                continue
            if part_type == "input_audio":
                _unsupported(f"{param}.{index}.type", "Audio input is not supported by Respawn.")
            _unsupported(f"{param}.{index}.type", f"Content part type '{part_type}' is not supported.")
        return
    if isinstance(content, dict):
        part_type = content.get("type")
        if part_type in TEXT_CONTENT_TYPES or part_type in MULTIMODAL_CONTENT_TYPES:
            return
        if part_type == "input_audio":
            _unsupported(f"{param}.type", "Audio input is not supported by Respawn.")
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


def _validate_stream_options(request: ResponseRequest) -> None:
    stream_options = request.stream_options
    if stream_options is None:
        return
    if not request.stream:
        raise OpenAIError("stream_options can only be set when stream=true.", param="stream_options", code="invalid_request")
    if not isinstance(stream_options, dict):
        raise OpenAIError("stream_options must be an object.", param="stream_options")
    for key in stream_options:
        if key != "include_obfuscation":
            _unsupported(f"stream_options.{key}", f"Stream option '{key}' is not supported.")
    include_obfuscation = stream_options.get("include_obfuscation")
    if include_obfuscation is not None and not isinstance(include_obfuscation, bool):
        raise OpenAIError("stream_options.include_obfuscation must be a boolean.", param="stream_options.include_obfuscation")


def _validate_metadata(metadata: dict[str, Any]) -> None:
    if not isinstance(metadata, dict):
        raise OpenAIError("metadata must be an object.", param="metadata")
    if len(metadata) > METADATA_MAX_PAIRS:
        raise OpenAIError(f"metadata must contain at most {METADATA_MAX_PAIRS} key-value pairs.", param="metadata")
    for key, value in metadata.items():
        if len(str(key)) > METADATA_KEY_MAX_LENGTH:
            raise OpenAIError(f"metadata keys must be at most {METADATA_KEY_MAX_LENGTH} characters.", param=f"metadata.{key}")
        if not isinstance(value, str):
            raise OpenAIError("metadata values must be strings.", param=f"metadata.{key}")
        if len(value) > METADATA_VALUE_MAX_LENGTH:
            raise OpenAIError(f"metadata values must be at most {METADATA_VALUE_MAX_LENGTH} characters.", param=f"metadata.{key}")


def _validate_service_tier(service_tier: str | None) -> None:
    if service_tier is not None and service_tier not in SUPPORTED_SERVICE_TIERS:
        _unsupported("service_tier", f"Service tier '{service_tier}' is not supported.")


def _validate_function_tool_protocol(request: ResponseRequest) -> None:
    names = []
    for normalized in _normalized_function_tools(request.tools):
        names.append(normalized["function"]["name"])
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise OpenAIError(f"Duplicate function tool name '{duplicates[0]}'.", param="tools", code="invalid_tool")

    _validate_web_search_tools(request)
    _validate_image_generation_tools(request)
    _validate_tool_choice(
        request.tool_choice,
        names,
        web_search_present=web_search_requested(request),
        image_generation_present=image_generation_requested(request),
        explicit="tool_choice" in request.model_fields_set,
    )
    if request.max_tool_calls is not None and request.max_tool_calls < 0:
        raise OpenAIError("max_tool_calls must be greater than or equal to 0.", param="max_tool_calls")
    if request.max_tool_calls == 0 and _tool_choice_requires_call(request.tool_choice):
        raise OpenAIError("max_tool_calls=0 conflicts with a required tool choice.", param="max_tool_calls", code="invalid_request")


def _normalized_function_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        param = f"tools.{index}"
        if not isinstance(tool, dict):
            raise OpenAIError("tools entries must be objects.", param=param)
        tool_type = tool.get("type")
        if tool_type == "function":
            normalized.append(_normalized_function_tool_body(tool, param=param))
            continue
        if tool_type == "namespace":
            normalized.extend(_normalized_namespace_tools(tool, param=param))
            continue
        if tool_type in WEB_SEARCH_TOOL_TYPES:
            continue
        if tool_type == IMAGE_GENERATION_TOOL_TYPE:
            continue
        FUNCTION_TOOL_UNSUPPORTED.labels(category=str(tool_type or "missing")).inc()
        _unsupported(f"{param}.type", f"Tool type '{tool_type}' is not supported. Respawn supports function tools as protocol data only.")
    return normalized


def _normalized_namespace_tools(tool: dict[str, Any], *, param: str) -> list[dict[str, Any]]:
    namespace = tool.get("name")
    _validate_tool_name(namespace, param=f"{param}.name")
    description = tool.get("description")
    if description is not None and not isinstance(description, str):
        raise OpenAIError("Namespace tool description must be a string.", param=f"{param}.description")
    tools = tool.get("tools", [])
    if not isinstance(tools, list):
        raise OpenAIError("Namespace tool tools must be a list.", param=f"{param}.tools")
    return [
        _normalized_function_tool_body(inner_tool, param=f"{param}.tools.{index}", namespace=str(namespace))
        for index, inner_tool in enumerate(tools)
    ]


def _normalized_function_tool_body(tool: dict[str, Any], *, param: str, namespace: str | None = None) -> dict[str, Any]:
    if not isinstance(tool, dict):
        raise OpenAIError("tools entries must be objects.", param=param)
    tool_type = tool.get("type")
    if tool_type != "function":
        FUNCTION_TOOL_UNSUPPORTED.labels(category=str(tool_type or "missing")).inc()
        _unsupported(f"{param}.type", f"Tool type '{tool_type}' is not supported. Respawn supports function tools as protocol data only.")

    function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    name = function.get("name")
    _validate_tool_name(name, param=f"{param}.name")
    description = function.get("description")
    if description is not None and not isinstance(description, str):
        raise OpenAIError("Function tool description must be a string.", param=f"{param}.description")
    parameters = function.get("parameters", {"type": "object", "properties": {}})
    if not isinstance(parameters, dict):
        raise OpenAIError("Function tool parameters must be a JSON Schema object.", param=f"{param}.parameters")
    try:
        Draft202012Validator.check_schema(parameters)
    except SchemaError as exc:
        raise OpenAIError("Function tool parameters must be a valid JSON Schema.", param=f"{param}.parameters", code="invalid_tool_schema") from exc
    strict = function.get("strict", tool.get("strict"))
    if strict is not None and not isinstance(strict, bool):
        raise OpenAIError("Function tool strict must be a boolean.", param=f"{param}.strict")

    original_name = str(name)
    backend_name = _flat_tool_name(namespace, original_name)
    normalized_function: dict[str, Any] = {"name": backend_name, "parameters": parameters}
    if description is not None:
        normalized_function["description"] = description
    if strict is not None:
        normalized_function["strict"] = strict
    normalized = {"type": "function", "function": normalized_function, "name": backend_name, "original_name": original_name}
    if namespace is not None:
        normalized["namespace"] = namespace
    return normalized


def _backend_function_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": dict(tool["function"]), "name": str(tool["function"]["name"])}


def _backend_web_search_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": WEB_SEARCH_INTERNAL_TOOL_NAME,
        "function": {
            "name": WEB_SEARCH_INTERNAL_TOOL_NAME,
            "description": (
                "Search the web for current, recent, external, or source-backed information. "
                "Use this tool when the answer needs online information or citations."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise web search query for the requested information.",
                    }
                },
                "required": ["query"],
            },
        },
    }


def _backend_image_generation_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": IMAGE_GENERATION_INTERNAL_TOOL_NAME,
        "function": {
            "name": IMAGE_GENERATION_INTERNAL_TOOL_NAME,
            "description": (
                "Generate or revise an image only when the latest user intent is to create or modify "
                "visual output. Consider immediate prior image outputs for short visual feedback, but "
                "do not infer image generation from history alone. When the latest request is not "
                "clearly visual, answer in text instead of calling this tool."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A concise image prompt for the image model. Translate non-English requests to English when helpful.",
                    }
                },
                "required": ["prompt"],
            },
        },
    }


def _include_internal_image_generation_tool(request: ResponseRequest) -> bool:
    if not image_generation_requested(request) or image_generation_disabled_by_choice(request):
        return False
    if image_generation_required(request):
        return False
    if isinstance(request.tool_choice, dict) and request.tool_choice.get("type") == "allowed_tools":
        return False
    return True


def _include_internal_web_search_tool(request: ResponseRequest) -> bool:
    if not web_search_requested(request) or web_search_disabled_by_choice(request):
        return False
    if web_search_required(request):
        return False
    if isinstance(request.tool_choice, dict) and request.tool_choice.get("type") == "allowed_tools":
        return False
    return True


def _validate_tool_choice(tool_choice: str | dict[str, Any] | None, names: list[str], *, web_search_present: bool, image_generation_present: bool, explicit: bool) -> None:
    if tool_choice is None:
        return
    if isinstance(tool_choice, str):
        if tool_choice not in SUPPORTED_TOOL_CHOICE_STRINGS:
            _unsupported("tool_choice", f"Tool choice '{tool_choice}' is not supported.")
        if explicit and tool_choice in {"auto", "required"} and not names and not web_search_present and not image_generation_present:
            raise OpenAIError("tool_choice requires at least one tool.", param="tools", code="invalid_request")
        return
    if not isinstance(tool_choice, dict):
        raise OpenAIError("tool_choice must be a string or object.", param="tool_choice")
    choice_type = tool_choice.get("type")
    if choice_type == "web_search":
        if not web_search_present:
            raise OpenAIError("tool_choice references an unknown web_search tool.", param="tool_choice.type", code="invalid_tool_choice")
        return
    if choice_type == IMAGE_GENERATION_TOOL_TYPE:
        if not image_generation_present:
            raise OpenAIError("tool_choice references an unknown image_generation tool.", param="tool_choice.type", code="invalid_tool_choice")
        return
    if choice_type == "function":
        name = _tool_choice_function_name(tool_choice)
        _validate_tool_name(name, param="tool_choice.name")
        namespace = _tool_choice_function_namespace(tool_choice)
        if namespace is not None:
            _validate_tool_name(namespace, param="tool_choice.namespace")
        if _tool_choice_backend_function_name(tool_choice) not in names:
            raise OpenAIError("tool_choice references an unknown function tool.", param="tool_choice.name", code="invalid_tool_choice")
        return
    if choice_type == "allowed_tools":
        mode = tool_choice.get("mode", "auto")
        if mode not in {"auto", "required"}:
            _unsupported("tool_choice.mode", f"Allowed tools mode '{mode}' is not supported.")
        tools = tool_choice.get("tools")
        if not isinstance(tools, list) or not tools:
            raise OpenAIError("tool_choice.allowed_tools requires a non-empty tools list.", param="tool_choice.tools")
        for index, tool in enumerate(tools):
            if not isinstance(tool, dict) or tool.get("type") != "function":
                _unsupported(f"tool_choice.tools.{index}.type", "Only function tools can be listed in allowed_tools.")
            name = tool.get("name")
            _validate_tool_name(name, param=f"tool_choice.tools.{index}.name")
            namespace = _tool_choice_function_namespace(tool)
            if namespace is not None:
                _validate_tool_name(namespace, param=f"tool_choice.tools.{index}.namespace")
            if _tool_choice_backend_function_name(tool) not in names:
                raise OpenAIError("allowed_tools references an unknown function tool.", param=f"tool_choice.tools.{index}.name", code="invalid_tool_choice")
        return
    _unsupported("tool_choice.type", f"Tool choice type '{choice_type}' is not supported.")


def _tool_choice_function_name(tool_choice: dict[str, Any]) -> str:
    function = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else {}
    return str(tool_choice.get("name") or function.get("name") or "")


def _tool_choice_function_namespace(tool_choice: dict[str, Any]) -> str | None:
    function = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else {}
    namespace = tool_choice.get("namespace") or function.get("namespace")
    return str(namespace) if namespace is not None else None


def _tool_choice_backend_function_name(tool_choice: dict[str, Any]) -> str:
    return _flat_tool_name(_tool_choice_function_namespace(tool_choice), _tool_choice_function_name(tool_choice))


def _tool_choice_requires_call(tool_choice: str | dict[str, Any] | None) -> bool:
    if tool_choice == "required":
        return True
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return True
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "allowed_tools" and tool_choice.get("mode") == "required":
        return True
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "web_search":
        return True
    if isinstance(tool_choice, dict) and tool_choice.get("type") == IMAGE_GENERATION_TOOL_TYPE:
        return True
    return False


def _validate_web_search_tools(request: ResponseRequest) -> None:
    for index, tool in enumerate(request.tools):
        if isinstance(tool, dict) and tool.get("type") in WEB_SEARCH_TOOL_TYPES:
            _normalized_web_search_tool(tool, param=f"tools.{index}")


def _normalized_web_search_tool(tool: dict[str, Any], *, param: str) -> dict[str, Any]:
    unsupported_keys = sorted(set(tool) - WEB_SEARCH_TOOL_FIELDS - {"return_token_budget"})
    if unsupported_keys:
        _unsupported(f"{param}.{unsupported_keys[0]}", f"web_search field '{unsupported_keys[0]}' is not supported.")
    if "return_token_budget" in tool:
        _unsupported(f"{param}.return_token_budget", "web_search return_token_budget is not supported yet.")

    context_size = tool.get("search_context_size", "medium")
    if context_size not in WEB_SEARCH_CONTEXT_SIZES:
        _unsupported(f"{param}.search_context_size", "web_search search_context_size must be one of low, medium, or high.")

    filters = tool.get("filters") or {}
    if not isinstance(filters, dict):
        raise OpenAIError("web_search filters must be an object.", param=f"{param}.filters")
    unsupported_filters = sorted(set(filters) - WEB_SEARCH_FILTER_FIELDS)
    if unsupported_filters:
        _unsupported(f"{param}.filters.{unsupported_filters[0]}", f"web_search filter '{unsupported_filters[0]}' is not supported.")

    external_web_access = tool.get("external_web_access", True)
    if not isinstance(external_web_access, bool):
        raise OpenAIError("web_search external_web_access must be a boolean.", param=f"{param}.external_web_access")

    user_location = tool.get("user_location")
    if user_location is not None and not isinstance(user_location, dict):
        raise OpenAIError("web_search user_location must be an object.", param=f"{param}.user_location")

    return {
        "type": "web_search",
        "search_context_size": context_size,
        "allowed_domains": _validated_domains(filters.get("allowed_domains", []), param=f"{param}.filters.allowed_domains"),
        "blocked_domains": _validated_domains(filters.get("blocked_domains", []), param=f"{param}.filters.blocked_domains"),
        "external_web_access": external_web_access,
        "user_location": user_location,
        "_param": param,
    }


def _validate_image_generation_tools(request: ResponseRequest) -> None:
    for index, tool in enumerate(request.tools):
        if isinstance(tool, dict) and tool.get("type") == IMAGE_GENERATION_TOOL_TYPE:
            _normalized_image_generation_tool(tool, param=f"tools.{index}")


def _normalized_image_generation_tool(tool: dict[str, Any], *, param: str) -> dict[str, Any]:
    unsupported_keys = sorted(set(tool) - IMAGE_GENERATION_TOOL_FIELDS - {"compression"})
    if unsupported_keys:
        _unsupported(f"{param}.{unsupported_keys[0]}", f"image_generation field '{unsupported_keys[0]}' is not supported.")
    if "compression" in tool:
        _unsupported(f"{param}.compression", "image_generation compression is not supported yet.")

    size = str(tool.get("size") or "auto")
    if size != "auto":
        match = IMAGE_GENERATION_SIZE_RE.fullmatch(size)
        if match is None:
            raise OpenAIError("image_generation size must be 'auto' or WIDTHxHEIGHT.", param=f"{param}.size", code="invalid_request")
        width = int(match.group(1))
        height = int(match.group(2))
    else:
        width = height = None

    quality = str(tool.get("quality") or "auto")
    if quality not in IMAGE_GENERATION_QUALITIES:
        _unsupported(f"{param}.quality", "image_generation quality must be one of auto, low, medium, or high.")

    output_format = str(tool.get("output_format") or "png").lower()
    if output_format not in IMAGE_GENERATION_OUTPUT_FORMATS:
        _unsupported(f"{param}.output_format", "Only png image_generation output_format is supported.")

    background = str(tool.get("background") or "auto")
    if background not in IMAGE_GENERATION_BACKGROUNDS:
        _unsupported(f"{param}.background", "Only auto or opaque image_generation background is supported.")

    action = str(tool.get("action") or "auto")
    if action not in IMAGE_GENERATION_ACTIONS:
        _unsupported(f"{param}.action", "Only auto or generate image_generation action is supported.")

    partial_images = tool.get("partial_images", 0)
    if not isinstance(partial_images, int) or partial_images < 0:
        raise OpenAIError("image_generation partial_images must be an integer.", param=f"{param}.partial_images")
    if partial_images > 0:
        _unsupported(f"{param}.partial_images", "image_generation partial_images streaming is not supported yet.")

    moderation = tool.get("moderation")
    if moderation is not None and not isinstance(moderation, str | bool):
        raise OpenAIError("image_generation moderation must be a string or boolean.", param=f"{param}.moderation")

    return {
        "type": IMAGE_GENERATION_TOOL_TYPE,
        "size": size,
        "width": width,
        "height": height,
        "quality": quality,
        "output_format": output_format,
        "background": background,
        "partial_images": partial_images,
        "action": action,
        "moderation": moderation,
        "_param": param,
    }


def _validated_domains(value: Any, *, param: str) -> list[str]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise OpenAIError("web_search domain filters must be lists.", param=param)
    if len(value) > WEB_SEARCH_MAX_DOMAINS:
        raise OpenAIError(f"web_search domain filters may contain at most {WEB_SEARCH_MAX_DOMAINS} domains.", param=param)
    domains: list[str] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, str):
            raise OpenAIError("web_search domain filters must contain strings.", param=f"{param}.{index}")
        domain = _normalize_domain(raw)
        if domain is None:
            raise OpenAIError("web_search domain filters must be domains without scheme, path, port, or wildcard.", param=f"{param}.{index}", code="invalid_request")
        if domain not in domains:
            domains.append(domain)
    return domains


def _normalize_domain(value: str) -> str | None:
    domain = value.strip().lower().rstrip(".")
    if not domain or "://" in domain or "/" in domain or ":" in domain or "*" in domain:
        return None
    if len(domain) > 253:
        return None
    labels = domain.split(".")
    if any(not label or len(label) > 63 for label in labels):
        return None
    if not all(DOMAIN_RE.fullmatch(label) for label in labels):
        return None
    return domain


def _validate_function_call_item(item: dict[str, Any], *, param: str) -> None:
    if not item.get("call_id"):
        raise OpenAIError("function_call input items require call_id.", param=f"{param}.call_id")
    _validate_tool_name(item.get("name"), param=f"{param}.name")
    if item.get("namespace") is not None:
        _validate_tool_name(item.get("namespace"), param=f"{param}.namespace")
    arguments = item.get("arguments", "{}")
    arguments_string = _arguments_to_string(arguments)
    _validate_json_arguments(arguments_string, param=f"{param}.arguments")


def _validate_function_call_output_item(item: dict[str, Any], *, param: str) -> None:
    if not item.get("call_id"):
        raise OpenAIError("function_call_output input items require call_id.", param=f"{param}.call_id")
    if "output" not in item:
        raise OpenAIError("function_call_output input items require output.", param=f"{param}.output")


def _validate_protocol_input_item(item: dict[str, Any], *, param: str) -> None:
    item_type = item.get("type")
    if item_type == "web_search_call":
        action = item.get("action")
        if action is not None and not isinstance(action, dict):
            raise OpenAIError("web_search_call input items require action to be an object.", param=f"{param}.action")
        return
    if item_type == "image_generation_call":
        if item.get("result") is not None and not isinstance(item.get("result"), str):
            raise OpenAIError("image_generation_call input items require result to be a string.", param=f"{param}.result")
        return


def _validate_reasoning_item(item: dict[str, Any], *, param: str) -> None:
    summary = item.get("summary", [])
    if summary is not None and not isinstance(summary, list):
        raise OpenAIError("reasoning summary must be a list.", param=f"{param}.summary")
    for index, part in enumerate(summary or []):
        if not isinstance(part, dict):
            raise OpenAIError("reasoning summary entries must be objects.", param=f"{param}.summary.{index}")
        if part.get("type") != "summary_text":
            _unsupported(f"{param}.summary.{index}.type", f"Reasoning summary type '{part.get('type')}' is not supported.")
        if not isinstance(part.get("text"), str):
            raise OpenAIError("reasoning summary text must be a string.", param=f"{param}.summary.{index}.text")

    encrypted_content = item.get("encrypted_content")
    if encrypted_content is not None and not isinstance(encrypted_content, str):
        raise OpenAIError("reasoning encrypted_content must be a string.", param=f"{param}.encrypted_content")

    unsupported_keys = sorted(set(item) - {"id", "type", "summary", "encrypted_content", "status"})
    if unsupported_keys:
        _unsupported(f"{param}.{unsupported_keys[0]}", f"Reasoning item field '{unsupported_keys[0]}' is not supported.")


def _validate_tool_name(name: Any, *, param: str) -> None:
    if not isinstance(name, str) or not FUNCTION_TOOL_NAME_RE.fullmatch(name):
        raise OpenAIError("Function tool names must be 1-64 characters and contain only letters, numbers, underscores, and hyphens.", param=param, code="invalid_tool")


def _arguments_to_string(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)


def _validate_json_arguments(arguments: str, *, param: str) -> None:
    try:
        json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        raise OpenAIError("function_call arguments must be a JSON string.", param=param, code="invalid_tool_arguments") from exc


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


def _flat_tool_name(namespace: str | None, name: str) -> str:
    return f"{namespace}{name}" if namespace else name


def _canonical_flat_tool_name(name: str) -> str:
    return re.sub(r"[:./-]+", "__", name)


def _function_tool_call_lookup(request: ResponseRequest) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for tool in _normalized_function_tools(request.tools):
        entry = {"name": str(tool.get("original_name") or tool["function"]["name"])}
        if tool.get("namespace") is not None:
            entry["namespace"] = str(tool["namespace"])
        lookup[str(tool["function"]["name"])] = entry
    return lookup
