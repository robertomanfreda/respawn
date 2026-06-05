from typing import Any

from src.schemas.errors import OpenAIError


def build_messages(
    *,
    instructions: str | None,
    chain: list[dict[str, Any]],
    input_value: str | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    for response in chain:
        request = response["request_json"]
        messages.extend(input_to_messages(request.get("input")))
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
    for item in input_value:
        item_type = item.get("type")
        role = item.get("role")
        if item_type == "message" or role in {"user", "assistant", "system", "developer"}:
            mapped_role = "system" if role == "developer" else role or "user"
            messages.append({"role": mapped_role, "content": content_to_text(item.get("content", ""))})
        elif item_type == "reasoning":
            continue
        elif item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id"),
                            "type": "function",
                            "function": {"name": item.get("name"), "arguments": item.get("arguments", "{}")},
                        }
                    ],
                }
            )
        elif item_type in {"function_call_output", "tool_result"}:
            messages.append({"role": "tool", "tool_call_id": item.get("call_id") or item.get("tool_call_id"), "content": content_to_text(item.get("output", ""))})
        else:
            messages.append({"role": "user", "content": content_to_text(item.get("content", item))})
    return messages


def output_to_messages(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in output or []:
        if item.get("type") == "message":
            messages.append({"role": item.get("role", "assistant"), "content": content_to_text(item.get("content", ""))})
        elif item.get("type") == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id"),
                            "type": "function",
                            "function": {"name": item.get("name"), "arguments": item.get("arguments", "{}")},
                        }
                    ],
                }
            )
    return messages


def response_tools_to_chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function":
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
    return converted or None


def assistant_text_to_output(message_id: str, text: str) -> dict[str, Any]:
    return {
        "id": message_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def tool_call_to_output(call: dict[str, Any]) -> dict[str, Any]:
    fn = call.get("function", {})
    return {
        "id": call.get("id"),
        "type": "function_call",
        "call_id": call.get("id"),
        "name": fn.get("name"),
        "arguments": fn.get("arguments", "{}"),
        "status": "completed",
    }


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", part.get("output_text", ""))))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    if isinstance(content, dict):
        return str(content.get("text", content))
    return str(content)
