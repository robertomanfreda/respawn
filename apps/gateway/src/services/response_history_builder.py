from typing import Any

import json

from src.schemas.errors import OpenAIError


TOOL_ITEM_TYPES = {"function_call", "function_call_output", "tool_result"}


def build_messages(
    *,
    instructions: str | None,
    chain: list[dict[str, Any]],
    input_value: str | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    for response in chain:
        request = response["request_json"]
        messages.extend(input_to_messages(response.get("input_items") or request.get("input")))
        messages.extend(output_to_messages(response["output_json"]))

    messages.extend(input_to_messages(input_value))
    return messages


def input_to_messages(input_value: str | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if input_value is None:
        return []
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]
    if not isinstance(input_value, list):
        raise OpenAIError("input must be a string or list of input items.", param="input")

    messages: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []
    for item in input_value:
        item_type = item.get("type")
        role = item.get("role")
        if item_type == "message" or role in {"user", "assistant", "system", "developer"}:
            _flush_tool_calls(messages, pending_tool_calls)
            mapped_role = "system" if role == "developer" else role or "user"
            messages.append({"role": mapped_role, "content": content_to_message_content(item.get("content", ""))})
        elif item_type == "reasoning":
            continue
        elif item_type == "function_call":
            pending_tool_calls.append(_chat_tool_call(item))
        elif item_type == "function_call_output":
            _flush_tool_calls(messages, pending_tool_calls)
            messages.append({"role": "tool", "tool_call_id": str(item["call_id"]), "content": function_output_to_text(item.get("output", ""))})
        elif item_type == "tool_result":
            continue
        else:
            _flush_tool_calls(messages, pending_tool_calls)
            messages.append({"role": "user", "content": content_to_message_content(item.get("content", item))})
    _flush_tool_calls(messages, pending_tool_calls)
    return messages


def output_to_messages(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []
    for item in output or []:
        if item.get("type") == "message":
            _flush_tool_calls(messages, pending_tool_calls)
            messages.append({"role": item.get("role", "assistant"), "content": content_to_text(item.get("content", ""))})
        elif item.get("type") == "function_call":
            pending_tool_calls.append(_chat_tool_call(item))
        elif item.get("type") in {"function_call_output", "tool_result", "reasoning"}:
            continue
    _flush_tool_calls(messages, pending_tool_calls)
    return messages


def assistant_text_to_output(message_id: str, text: str) -> dict[str, Any]:
    return {
        "id": message_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
    }


def _chat_tool_call(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(item["call_id"]),
        "type": "function",
        "function": {
            "name": str(item["name"]),
            "arguments": _arguments_to_string(item.get("arguments", "{}")),
        },
    }


def _flush_tool_calls(messages: list[dict[str, Any]], pending_tool_calls: list[dict[str, Any]]) -> None:
    if not pending_tool_calls:
        return
    messages.append({"role": "assistant", "content": "", "tool_calls": list(pending_tool_calls)})
    pending_tool_calls.clear()


def function_output_to_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, separators=(",", ":"), ensure_ascii=False)


def _arguments_to_string(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)


def content_to_message_content(content: Any) -> Any:
    if isinstance(content, list) and any(isinstance(part, dict) and part.get("type") in {"input_image", "input_file"} for part in content):
        return [dict(part) if isinstance(part, dict) else {"type": "input_text", "text": str(part)} for part in content]
    if isinstance(content, dict) and content.get("type") in {"input_image", "input_file"}:
        return [dict(content)]
    return content_to_text(content)


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "input_file":
                    filename = part.get("filename") or "input_file"
                    file_text = part.get("text") or part.get("extracted_text") or ""
                    parts.append(f"[File: {filename}]\n{file_text}".strip())
                elif part_type == "input_image":
                    parts.append("[Image input]")
                else:
                    parts.append(str(part.get("text", part.get("output_text", ""))))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    if isinstance(content, dict):
        if content.get("type") == "input_file":
            filename = content.get("filename") or "input_file"
            file_text = content.get("text") or content.get("extracted_text") or ""
            return f"[File: {filename}]\n{file_text}".strip()
        if content.get("type") == "input_image":
            return "[Image input]"
        return str(content.get("text", content))
    return str(content)
