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
SUPPORTED_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
SUPPORTED_REASONING_SUMMARIES = {"auto", "concise", "detailed"}
SUPPORTED_PROMPT_CACHE_RETENTIONS = {"in_memory", "24h"}
SUPPORTED_SERVICE_TIERS = {"auto", "default", "flex", "scale", "priority"}
SUPPORTED_TOOL_CHOICE_STRINGS = {"auto", "none", "required"}
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
    tools = [_normalized_function_tool(tool, param=f"tools.{index}") for index, tool in enumerate(request.tools)]
    tool_choice = request.tool_choice
    if isinstance(tool_choice, str) or tool_choice is None:
        return tools
    if not isinstance(tool_choice, dict) or tool_choice.get("type") != "allowed_tools":
        return tools
    allowed_names = {tool["name"] for tool in tool_choice.get("tools") or [] if isinstance(tool, dict)}
    return [tool for tool in tools if tool["function"]["name"] in allowed_names]


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
        name = _tool_choice_function_name(tool_choice)
        return {"type": "function", "function": {"name": name}}
    if choice_type == "allowed_tools":
        return tool_choice.get("mode", "auto")
    return None


def tool_choice_instruction(request: ResponseRequest) -> str | None:
    if not request.tools:
        return None
    tool_choice = request.tool_choice
    if tool_choice == "required":
        return "You must call one of the provided functions instead of answering in natural language."
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return f"You must call the function named {_tool_choice_function_name(tool_choice)} instead of answering in natural language."
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "allowed_tools" and tool_choice.get("mode") == "required":
        names = ", ".join(tool.get("name", "") for tool in tool_choice.get("tools") or [] if isinstance(tool, dict))
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
    return {
        "id": str(item.get("id") or fallback_id),
        "type": "function_call",
        "call_id": str(item["call_id"]),
        "name": str(item["name"]),
        "arguments": _arguments_to_string(item.get("arguments", "{}")),
        "status": item.get("status", "completed"),
    }


def _function_call_output_item(item: dict[str, Any], *, fallback_id: str) -> dict[str, Any]:
    return function_call_output_item(
        str(item.get("id") or fallback_id),
        call_id=str(item["call_id"]),
        output=item.get("output", ""),
        status=item.get("status", "completed"),
    )


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
    for index, tool in enumerate(request.tools):
        normalized = _normalized_function_tool(tool, param=f"tools.{index}")
        names.append(normalized["function"]["name"])
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise OpenAIError(f"Duplicate function tool name '{duplicates[0]}'.", param="tools", code="invalid_tool")

    _validate_tool_choice(request.tool_choice, names, explicit="tool_choice" in request.model_fields_set)
    if request.max_tool_calls is not None and request.max_tool_calls < 0:
        raise OpenAIError("max_tool_calls must be greater than or equal to 0.", param="max_tool_calls")
    if request.max_tool_calls == 0 and _tool_choice_requires_call(request.tool_choice):
        raise OpenAIError("max_tool_calls=0 conflicts with a required tool choice.", param="max_tool_calls", code="invalid_request")


def _normalized_function_tool(tool: dict[str, Any], *, param: str) -> dict[str, Any]:
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

    normalized_function: dict[str, Any] = {"name": str(name), "parameters": parameters}
    if description is not None:
        normalized_function["description"] = description
    if strict is not None:
        normalized_function["strict"] = strict
    return {"type": "function", "function": normalized_function, "name": str(name)}


def _validate_tool_choice(tool_choice: str | dict[str, Any] | None, names: list[str], *, explicit: bool) -> None:
    if tool_choice is None:
        return
    if isinstance(tool_choice, str):
        if tool_choice not in SUPPORTED_TOOL_CHOICE_STRINGS:
            _unsupported("tool_choice", f"Tool choice '{tool_choice}' is not supported.")
        if explicit and tool_choice in {"auto", "required"} and not names:
            raise OpenAIError("tool_choice requires at least one function tool.", param="tools", code="invalid_request")
        return
    if not isinstance(tool_choice, dict):
        raise OpenAIError("tool_choice must be a string or object.", param="tool_choice")
    choice_type = tool_choice.get("type")
    if choice_type == "function":
        name = _tool_choice_function_name(tool_choice)
        _validate_tool_name(name, param="tool_choice.name")
        if name not in names:
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
            if name not in names:
                raise OpenAIError("allowed_tools references an unknown function tool.", param=f"tool_choice.tools.{index}.name", code="invalid_tool_choice")
        return
    _unsupported("tool_choice.type", f"Tool choice type '{choice_type}' is not supported.")


def _tool_choice_function_name(tool_choice: dict[str, Any]) -> str:
    function = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else {}
    return str(tool_choice.get("name") or function.get("name") or "")


def _tool_choice_requires_call(tool_choice: str | dict[str, Any] | None) -> bool:
    if tool_choice == "required":
        return True
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return True
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "allowed_tools" and tool_choice.get("mode") == "required":
        return True
    return False


def _validate_function_call_item(item: dict[str, Any], *, param: str) -> None:
    if not item.get("call_id"):
        raise OpenAIError("function_call input items require call_id.", param=f"{param}.call_id")
    _validate_tool_name(item.get("name"), param=f"{param}.name")
    arguments = item.get("arguments", "{}")
    arguments_string = _arguments_to_string(arguments)
    _validate_json_arguments(arguments_string, param=f"{param}.arguments")


def _validate_function_call_output_item(item: dict[str, Any], *, param: str) -> None:
    if not item.get("call_id"):
        raise OpenAIError("function_call_output input items require call_id.", param=f"{param}.call_id")
    if "output" not in item:
        raise OpenAIError("function_call_output input items require output.", param=f"{param}.output")


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
