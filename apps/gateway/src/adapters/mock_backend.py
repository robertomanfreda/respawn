from collections.abc import AsyncIterator
import asyncio
import json
from typing import Any

from src.adapters.base import ChatCompletionResult, ModelBackend
from src.adapters.mock_control import mock_options
from src.schemas.models import ModelList, ModelObject
from src.services.responses_compat import IMAGE_GENERATION_INTERNAL_TOOL_NAME, WEB_SEARCH_INTERNAL_TOOL_NAME
from src.services.structured_outputs import example_for_schema, schema_from_response_format


class MockBackend(ModelBackend):
    def __init__(self, default_model: str = "mock-model") -> None:
        self.default_model = default_model

    async def list_models(self) -> ModelList:
        return ModelList(data=[ModelObject(id=self.default_model, owned_by="mock")])

    async def create_chat_completion(self, payload: dict[str, Any]) -> ChatCompletionResult:
        messages = payload.get("messages", [])
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
        text = _content_to_text(last_user.get("content", ""))
        schema = schema_from_response_format(payload.get("response_format"))
        max_tokens = payload.get("max_tokens")
        tool_choice = payload.get("tool_choice")
        options = mock_options(payload.get("metadata"))

        delay_seconds = options.get("delay_seconds")
        if isinstance(delay_seconds, (int, float)) and delay_seconds > 0:
            await asyncio.sleep(float(delay_seconds))

        if schema:
            if options.get("structured_output") == "always_invalid":
                content = "not valid json"
                return ChatCompletionResult(content=content, usage=_usage(messages, content))
            if payload.get("_respawn_repair_attempt"):
                content = json.dumps(example_for_schema(schema), separators=(",", ":"))
                return ChatCompletionResult(content=content, usage=_usage(messages, content))
            content = "not valid json"
            return ChatCompletionResult(content=content, usage=_usage(messages, content))

        forced_tool_name = None
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            function = tool_choice.get("function") if isinstance(tool_choice.get("function"), dict) else {}
            forced_tool_name = function.get("name")

        requested_tool_call = options.get("tool_call")
        image_tool = _function_tool(payload.get("tools"), IMAGE_GENERATION_INTERNAL_TOOL_NAME)
        if image_tool and forced_tool_name in (None, IMAGE_GENERATION_INTERNAL_TOOL_NAME) and requested_tool_call == "image_generation":
            return ChatCompletionResult(
                content="",
                tool_calls=[
                    {
                        "id": "call_mock_image_generation",
                        "type": "function",
                        "function": {
                            "name": IMAGE_GENERATION_INTERNAL_TOOL_NAME,
                            "arguments": json.dumps({"prompt": text}, separators=(",", ":")),
                        },
                    }
                ],
                usage={"input_tokens": len(str(messages).split()), "output_tokens": 0, "total_tokens": len(str(messages).split())},
            )

        web_search_tool = _function_tool(payload.get("tools"), WEB_SEARCH_INTERNAL_TOOL_NAME)
        if web_search_tool and forced_tool_name in (None, WEB_SEARCH_INTERNAL_TOOL_NAME) and requested_tool_call == "web_search":
            return ChatCompletionResult(
                content="",
                tool_calls=[
                    {
                        "id": "call_mock_web_search",
                        "type": "function",
                        "function": {
                            "name": WEB_SEARCH_INTERNAL_TOOL_NAME,
                            "arguments": json.dumps({"query": text}, separators=(",", ":")),
                        },
                    }
                ],
                usage={"input_tokens": len(str(messages).split()), "output_tokens": 0, "total_tokens": len(str(messages).split())},
            )

        if payload.get("tools") and (tool_choice == "required" or forced_tool_name) and not any(m.get("role") == "tool" for m in messages):
            tool = next((candidate.get("function", {}) for candidate in payload["tools"] if candidate.get("function", {}).get("name") == forced_tool_name), None)
            if tool is None:
                tool = payload["tools"][0].get("function", {})
            tool_name = tool.get("name", "echo")
            arguments = json.dumps(example_for_schema(tool.get("parameters") or {"type": "object"}), separators=(",", ":"))
            return ChatCompletionResult(
                content="",
                tool_calls=[
                    {
                        "id": f"call_mock_{tool_name.replace('.', '_')}",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": arguments},
                    }
                ],
                usage={"input_tokens": len(str(messages).split()), "output_tokens": 0, "total_tokens": len(str(messages).split())},
            )

        if any(m.get("role") == "tool" for m in messages):
            tool_text = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "tool"), "")
            text = f"Tool result: {tool_text}"
        elif _has_image(messages):
            text = "The image contains a red square."
        elif _has_context_payload(messages) or options.get("include_context"):
            text = f"Mock response: {text}\nContext: {_context_text(messages)}"
        else:
            text = f"Mock response: {text}"

        finish_reason = None
        if max_tokens == 1:
            text = text.split()[0] if text.split() else ""
            finish_reason = "length"

        reasoning = ""
        if payload.get("reasoning"):
            reasoning = "The mock backend inspected the prompt and selected a direct response."

        out_tokens = len(text.split()) + len(reasoning.split())
        in_tokens = len(str(messages).split())
        usage: dict[str, Any] = {"input_tokens": in_tokens, "output_tokens": out_tokens, "total_tokens": in_tokens + out_tokens}
        if reasoning:
            usage["output_tokens_details"] = {"reasoning_tokens": len(reasoning.split())}
        return ChatCompletionResult(
            content=text,
            reasoning=reasoning,
            finish_reason=finish_reason,
            usage=usage,
            content_logprobs=_mock_logprobs(text, int(payload.get("top_logprobs") or 0)) if "top_logprobs" in payload else [],
        )

    async def create_chat_completion_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        result = await self.create_chat_completion(payload)
        if result.reasoning:
            for token in result.reasoning.split():
                yield {"type": "reasoning_delta", "delta": token + " "}
        if result.tool_calls:
            yield {"type": "tool_calls", "tool_calls": result.tool_calls}
        for token in result.content.split():
            yield {"type": "delta", "delta": token + " "}
        done = {"type": "done", "usage": result.usage}
        if result.finish_reason is not None:
            done["finish_reason"] = result.finish_reason
        yield done


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            part_type = part.get("type")
            if part_type == "input_file":
                parts.append(str(part.get("text", "")))
            elif part_type == "input_image":
                parts.append("[image]")
            else:
                parts.append(str(part.get("text", part.get("output_text", part))))
        return " ".join(parts)
    return str(content)


def _has_image(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(isinstance(part, dict) and part.get("type") == "input_image" for part in content):
            return True
    return False


def _has_context_payload(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") == "system":
            return True
        content = message.get("content")
        if isinstance(content, list) and any(isinstance(part, dict) and part.get("type") == "input_file" for part in content):
            return True
    return False


def _context_text(messages: list[dict[str, Any]]) -> str:
    text = " ".join(_content_to_text(message.get("content", "")) for message in messages)
    return text[:1200]


def _function_tool(tools: Any, name: str) -> dict[str, Any] | None:
    if not isinstance(tools, list):
        return None
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        if function.get("name") == name:
            return function
    return None


def _usage(messages: list[dict[str, Any]], content: str) -> dict[str, int]:
    out_tokens = len(content.split())
    in_tokens = len(str(messages).split())
    return {"input_tokens": in_tokens, "output_tokens": out_tokens, "total_tokens": in_tokens + out_tokens}


def _mock_logprobs(text: str, top_logprobs: int) -> list[dict[str, Any]]:
    entries = []
    for token in text.split():
        top = [{"token": token, "bytes": list(token.encode("utf-8")), "logprob": -0.01}]
        for index in range(max(top_logprobs, 0)):
            alternative = f"{token}_{index + 1}"
            top.append({"token": alternative, "bytes": list(alternative.encode("utf-8")), "logprob": -1.0 - index})
        entries.append({"token": token, "bytes": list(token.encode("utf-8")), "logprob": -0.01, "top_logprobs": top[: top_logprobs + 1]})
    return entries
