import json

import pytest

from src.streaming.events import make_event


IMPLEMENTED_RESPONSES_STREAM_EVENTS = [
    "response.created",
    "response.in_progress",
    "response.output_item.added",
    "response.content_part.added",
    "response.output_text.delta",
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary_text.done",
    "response.reasoning_summary_part.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.completed",
    "response.incomplete",
    "response.failed",
    "error",
]


@pytest.mark.parametrize("event_type", IMPLEMENTED_RESPONSES_STREAM_EVENTS)
def test_make_event_builds_valid_sse_for_implemented_events(event_type):
    rendered = make_event(event_type, {"ok": True}, sequence_number=7, event_id="evt_7")

    assert rendered.endswith("\n\n")
    assert rendered.startswith("id: evt_7\n")
    assert f"event: {event_type}\n" in rendered

    data_line = next(line for line in rendered.splitlines() if line.startswith("data: "))
    payload = json.loads(data_line.removeprefix("data: "))
    assert payload == {"type": event_type, "ok": True, "sequence_number": 7}
    assert "event_id" not in payload


def test_make_event_omits_sse_id_when_unset():
    rendered = make_event("response.completed", {"ok": True}, sequence_number=0)

    assert not rendered.startswith("id:")
    assert rendered.startswith("event: response.completed\n")
