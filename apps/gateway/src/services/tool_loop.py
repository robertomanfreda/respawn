import json
from typing import Any

from src.adapters.base import ChatCompletionResult, ModelBackend
from src.schemas.errors import OpenAIError
from src.services.conversation_builder import tool_call_to_output
from src.services.id_generator import generate_id
from src.storage.repository import ResponseRepository
from src.tools.registry import ToolRegistry


class ToolLoop:
    def __init__(
        self,
        backend: ModelBackend,
        registry: ToolRegistry,
        repository: ResponseRepository | None,
        max_iterations: int,
        tool_timeout_seconds: float,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.repository = repository
        self.max_iterations = max_iterations
        self.tool_timeout_seconds = tool_timeout_seconds

    async def run(self, *, response_id: str, payload: dict[str, Any]) -> ChatCompletionResult:
        result = await self.backend.create_chat_completion(payload)
        iterations = 0
        output_items: list[dict[str, Any]] = []
        usage = _merge_usage({}, result.usage)
        seen_call_ids: set[str] = set()
        while result.tool_calls and iterations < self.max_iterations:
            calls = [_normalize_tool_call(call, seen_call_ids) for call in result.tool_calls]
            if any(not self.registry.has(_tool_name(call)) for call in calls):
                result.output_items = [*output_items, *(tool_call_to_output(call) for call in calls)]
                result.usage = usage
                result.unhandled_tool_calls = True
                return result

            iterations += 1
            payload["messages"].append({"role": "assistant", "content": result.content or "", "tool_calls": calls})
            for call in calls:
                call_id = call["id"]
                fn = call.get("function", {})
                name = fn.get("name")
                arguments = fn.get("arguments") or "{}"
                output_items.append(tool_call_to_output(call))
                try:
                    output = await self.registry.execute(name, arguments, timeout=self.tool_timeout_seconds)
                    if self.repository is not None:
                        await self.repository.save_tool_call(
                            response_id=response_id,
                            call_id=call_id,
                            name=name,
                            arguments_json=json.loads(arguments or "{}"),
                            output_json=output,
                            status="completed",
                        )
                except Exception as exc:
                    if self.repository is not None:
                        await self.repository.save_tool_call(
                            response_id=response_id,
                            call_id=call_id,
                            name=name or "",
                            arguments_json=_safe_arguments(arguments),
                            output_json={"error": str(exc)},
                            status="failed",
                        )
                    raise
                payload["messages"].append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(output)})
            result = await self.backend.create_chat_completion(payload)
            usage = _merge_usage(usage, result.usage)
        if result.tool_calls:
            raise OpenAIError(
                "Tool call loop exceeded MAX_TOOL_ITERATIONS.",
                status_code=400,
                type="invalid_request_error",
                code="max_tool_iterations_exceeded",
            )
        result.usage = usage
        result.output_items = output_items
        return result


def _safe_arguments(arguments: str) -> Any:
    try:
        return json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {"raw": arguments}


def _normalize_tool_call(call: dict[str, Any], seen_call_ids: set[str]) -> dict[str, Any]:
    normalized = dict(call)
    call_id = normalized.get("id") or generate_id("call")
    while call_id in seen_call_ids:
        call_id = generate_id("call")
    seen_call_ids.add(call_id)
    normalized["id"] = call_id
    normalized.setdefault("type", "function")
    return normalized


def _tool_name(call: dict[str, Any]) -> str:
    fn = call.get("function") or {}
    return str(fn.get("name") or "")


def _merge_usage(left: dict[str, Any], right: dict[str, Any] | None) -> dict[str, Any]:
    right = right or {}
    right_input = int(right.get("input_tokens", right.get("prompt_tokens", 0)) or 0)
    right_output = int(right.get("output_tokens", right.get("completion_tokens", 0)) or 0)
    right_total = int(right.get("total_tokens", right_input + right_output) or 0)
    right_input_details = right.get("input_tokens_details") if isinstance(right.get("input_tokens_details"), dict) else {}
    right_output_details = right.get("output_tokens_details") if isinstance(right.get("output_tokens_details"), dict) else {}
    left_input = int(left.get("input_tokens", 0))
    left_output = int(left.get("output_tokens", 0))
    left_total = int(left.get("total_tokens", left_input + left_output) or 0)
    left_input_details = left.get("input_tokens_details") if isinstance(left.get("input_tokens_details"), dict) else {}
    left_output_details = left.get("output_tokens_details") if isinstance(left.get("output_tokens_details"), dict) else {}
    return {
        "input_tokens": left_input + right_input,
        "input_tokens_details": {"cached_tokens": int(left_input_details.get("cached_tokens", 0) or 0) + int(right_input_details.get("cached_tokens", 0) or 0)},
        "output_tokens": left_output + right_output,
        "output_tokens_details": {
            "reasoning_tokens": int(left_output_details.get("reasoning_tokens", 0) or 0) + int(right_output_details.get("reasoning_tokens", 0) or 0)
        },
        "total_tokens": left_total + right_total,
    }
