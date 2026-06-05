import json
from typing import Any


def make_event(event: str, data: dict[str, Any], sequence_number: int | None = None) -> str:
    payload = {"type": event, **data}
    if sequence_number is not None:
        payload["sequence_number"] = sequence_number
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
