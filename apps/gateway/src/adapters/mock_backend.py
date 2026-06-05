from collections.abc import AsyncIterator
import json
from typing import Any

from src.adapters.base import ChatCompletionResult, ModelBackend
from src.schemas.models import ModelList, ModelObject
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
        all_text = " ".join(_content_to_text(m.get("content", "")) for m in messages).lower()
        schema = schema_from_response_format(payload.get("response_format"))

        if schema:
            if "repair failure" in all_text:
                content = "not valid json"
                return ChatCompletionResult(content=content, usage=_usage(messages, content))
            if "previous assistant output was not valid" in all_text or "valid json" in all_text:
                content = json.dumps(example_for_schema(schema), separators=(",", ":"))
                return ChatCompletionResult(content=content, usage=_usage(messages, content))
            content = "not valid json"
            return ChatCompletionResult(content=content, usage=_usage(messages, content))

        if payload.get("tools") and "loop forever" in all_text:
            return ChatCompletionResult(
                content="",
                tool_calls=[
                    {
                        "id": "call_mock_loop",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"text":"again"}'},
                    }
                ],
                usage={"input_tokens": len(str(messages).split()), "output_tokens": 0, "total_tokens": len(str(messages).split())},
            )

        if payload.get("tools") and "repo browser" in all_text:
            tool = payload["tools"][0].get("function", {})
            return ChatCompletionResult(
                content="",
                tool_calls=[
                    {
                        "id": "call_mock_repo_browser",
                        "type": "function",
                        "function": {"name": tool.get("name", "repo_browser.list_files"), "arguments": '{"path":"."}'},
                    }
                ],
                usage={"input_tokens": len(str(messages).split()), "output_tokens": 0, "total_tokens": len(str(messages).split())},
            )

        if any(m.get("role") == "tool" for m in messages):
            tool_text = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "tool"), "")
            text = f"Tool result: {tool_text}"
        elif payload.get("tools") and "calculator" in text.lower():
            return ChatCompletionResult(
                content="",
                tool_calls=[
                    {
                        "id": "call_mock_calculator",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": '{"expression":"2+2"}'},
                    }
                ],
                usage={"input_tokens": len(str(messages).split()), "output_tokens": 0, "total_tokens": len(str(messages).split())},
            )

        else:
            text = f"Mock response: {text}"

        reasoning = ""
        if payload.get("reasoning"):
            reasoning = "The mock backend inspected the prompt and selected a direct response."

        out_tokens = len(text.split()) + len(reasoning.split())
        in_tokens = len(str(messages).split())
        usage: dict[str, Any] = {"input_tokens": in_tokens, "output_tokens": out_tokens, "total_tokens": in_tokens + out_tokens}
        if reasoning:
            usage["output_tokens_details"] = {"reasoning_tokens": len(reasoning.split())}
        return ChatCompletionResult(content=text, reasoning=reasoning, usage=usage)

    async def create_chat_completion_stream(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        result = await self.create_chat_completion(payload)
        if result.reasoning:
            for token in result.reasoning.split():
                yield {"type": "reasoning_delta", "delta": token + " "}
        for token in result.content.split():
            yield {"type": "delta", "delta": token + " "}
        yield {"type": "done", "usage": result.usage}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part.get("text", part)) for part in content)
    return str(content)


def _usage(messages: list[dict[str, Any]], content: str) -> dict[str, int]:
    out_tokens = len(content.split())
    in_tokens = len(str(messages).split())
    return {"input_tokens": in_tokens, "output_tokens": out_tokens, "total_tokens": in_tokens + out_tokens}
