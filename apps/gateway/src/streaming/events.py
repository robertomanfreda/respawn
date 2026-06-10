import json
from typing import Any


def make_event(event: str, data: dict[str, Any], sequence_number: int | None = None, event_id: str | None = None) -> str:
    payload = {"type": event, **data}
    if sequence_number is not None:
        payload["sequence_number"] = sequence_number
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(payload, separators=(',', ':'))}")
    return "\n".join(lines) + "\n\n"
