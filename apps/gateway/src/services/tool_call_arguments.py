from __future__ import annotations

import json
from typing import Any


def arguments_to_string(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)


def parse_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {}
    elif isinstance(arguments, dict):
        parsed = arguments
    else:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def tool_call_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    return parse_arguments(function.get("arguments", "{}"))
