import asyncio
import json
import operator
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from jsonschema import ValidationError, validate

from src.observability.metrics import TOOL_EXECUTIONS
from src.schemas.errors import OpenAIError


ToolHandler = Callable[[dict[str, Any]], Awaitable[Any] | Any]


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    schema: dict[str, Any]
    handler: ToolHandler


class ToolRegistry:
    """Small in-process registry for built-in function tools."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, name: str, schema: dict[str, Any], handler: ToolHandler) -> None:
        self._tools[name] = RegisteredTool(name=name, schema=schema, handler=handler)

    def has(self, name: str) -> bool:
        return name in self._tools

    async def execute(self, name: str, arguments_json: str, timeout: float = 15.0) -> Any:
        if name not in self._tools:
            raise OpenAIError(f"Tool '{name}' is not registered.", type="invalid_request_error", code="tool_not_found")
        tool = self._tools[name]
        try:
            arguments = json.loads(arguments_json or "{}")
            validate(instance=arguments, schema=tool.schema)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise OpenAIError(f"Invalid arguments for tool '{name}'.", type="invalid_request_error", code="invalid_tool_arguments") from exc

        try:
            result = tool.handler(arguments)
            if asyncio.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=timeout)
            TOOL_EXECUTIONS.labels(tool=name, status="completed").inc()
            return result
        except TimeoutError as exc:
            TOOL_EXECUTIONS.labels(tool=name, status="timeout").inc()
            raise OpenAIError(f"Tool '{name}' timed out.", status_code=500, type="server_error", code="tool_timeout") from exc
        except OpenAIError:
            TOOL_EXECUTIONS.labels(tool=name, status="failed").inc()
            raise
        except Exception as exc:
            TOOL_EXECUTIONS.labels(tool=name, status="failed").inc()
            raise OpenAIError(f"Tool '{name}' failed.", status_code=500, type="server_error", code="tool_execution_failed") from exc


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("echo", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}, lambda args: args["text"])
    registry.register("get_current_time", {"type": "object", "properties": {"timezone": {"type": "string"}}}, get_current_time)
    registry.register("calculator", {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}, calculate)
    return registry


def get_current_time(args: dict[str, Any]) -> dict[str, str]:
    from datetime import datetime, timezone

    return {"timezone": args.get("timezone", "UTC"), "iso": datetime.now(timezone.utc).isoformat()}


def calculate(args: dict[str, Any]) -> Any:
    import ast

    allowed_binops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow}
    allowed_unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_binops:
            return allowed_binops[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
            return allowed_unary[type(node.op)](eval_node(node.operand))
        raise OpenAIError("Unsupported calculator expression.", type="invalid_request_error", code="invalid_tool_arguments")

    return eval_node(ast.parse(args["expression"], mode="eval"))
