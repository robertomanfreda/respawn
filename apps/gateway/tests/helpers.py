from typing import Any


def backend_tool_names(payload: dict[str, Any]) -> list[str]:
    names = []
    for tool in payload.get("tools") or []:
        function = tool.get("function") if isinstance(tool, dict) and isinstance(tool.get("function"), dict) else {}
        names.append(function.get("name"))
    return names
