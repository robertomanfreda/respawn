import json

from src.adapters.mock_control import mock_metadata


def parse_sse_events(text):
    events = []
    for block in text.strip().split("\n\n"):
        if not block:
            continue
        event = {"id": None, "event": None, "data": None}
        for line in block.splitlines():
            if line.startswith("id: "):
                event["id"] = line.removeprefix("id: ").strip()
            elif line.startswith("event: "):
                event["event"] = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                event["data"] = json.loads(line.removeprefix("data: ").strip())
        events.append(event)
    return events


def test_streaming_event_lifecycle(client):
    with client.stream("POST", "/v1/responses", json={"input": "hello stream", "stream": True, "store": True}) as response:
        text = "".join(response.iter_text())
    events = parse_sse_events(text)
    event_types = [event["event"] for event in events]

    assert event_types[:2] == ["response.created", "response.in_progress"]
    assert "response.output_item.added" in event_types
    assert "response.content_part.added" in event_types
    assert "response.output_text.delta" in event_types
    assert "response.output_text.done" in event_types
    assert "response.content_part.done" in event_types
    assert "response.output_item.done" in event_types
    assert event_types[-1] == "response.completed"
    assert [event["data"]["sequence_number"] for event in events] == list(range(len(events)))
    assert all(event["id"] for event in events)
    assert all(event["data"]["type"] == event["event"] for event in events)
    metrics = client.get("/metrics").text
    assert 'gateway_streaming_responses_running{model="gpt-oss-120b"} 0.0' in metrics

    terminal_response = events[-1]["data"]["response"]
    retrieved = client.get(f"/v1/responses/{terminal_response['id']}").json()
    assert retrieved["status"] == terminal_response["status"]
    assert retrieved["output_text"] == terminal_response["output_text"]
    assert [item["id"] for item in retrieved["output"]] == [item["id"] for item in terminal_response["output"]]


def test_streaming_reasoning_events(client):
    with client.stream(
        "POST",
        "/v1/responses",
        json={"input": "hello reasoning stream", "stream": True, "reasoning": {"effort": "low", "summary": "auto"}},
    ) as response:
        text = "".join(response.iter_text())

    events = parse_sse_events(text)
    event_types = [event["event"] for event in events]
    assert "response.reasoning_summary_part.added" in event_types
    assert "response.reasoning_summary_text.delta" in event_types
    assert "response.reasoning_summary_text.done" in event_types
    assert "response.reasoning_summary_part.done" in event_types
    assert any((event["data"].get("item") or {}).get("type") == "reasoning" for event in events)
    assert any("Estimated reasoning tokens" in json.dumps(event["data"]) for event in events)
    assert event_types[-1] == "response.completed"


def test_streaming_incomplete_event(client):
    with client.stream("POST", "/v1/responses", json={"input": "please produce more than one token", "stream": True, "max_output_tokens": 1}) as response:
        text = "".join(response.iter_text())

    events = parse_sse_events(text)
    assert events[-1]["event"] == "response.incomplete"
    terminal_response = events[-1]["data"]["response"]
    assert terminal_response["status"] == "incomplete"
    assert terminal_response["incomplete_details"] == {"reason": "max_tokens"}


def test_streaming_failure_events(client):
    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "input": "stream structured output failure fixture",
            "stream": True,
            "text": {"format": {"type": "json_schema", "name": "impossible_schema", "schema": {"not": {}}}},
        },
    ) as response:
        text = "".join(response.iter_text())

    events = parse_sse_events(text)
    assert [event["event"] for event in events[-2:]] == ["response.failed", "error"]
    assert events[-2]["data"]["response"]["status"] == "failed"
    assert events[-1]["data"]["error"]["code"] == "structured_output_validation_failed"


def test_stream_options_disable_obfuscation(client):
    with client.stream(
        "POST",
        "/v1/responses",
        json={"input": "hello stream", "stream": True, "stream_options": {"include_obfuscation": False}},
    ) as response:
        text = "".join(response.iter_text())

    delta_events = [event for event in parse_sse_events(text) if event["event"] == "response.output_text.delta"]
    assert delta_events
    assert all("obfuscation" not in event["data"] for event in delta_events)


def test_stream_options_include_obfuscation_by_default(client):
    with client.stream("POST", "/v1/responses", json={"input": "hello stream", "stream": True}) as response:
        text = "".join(response.iter_text())

    delta_events = [event for event in parse_sse_events(text) if event["event"] == "response.output_text.delta"]
    assert delta_events
    assert all(isinstance(event["data"].get("obfuscation"), str) for event in delta_events)


def test_streaming_function_call_argument_events(client):
    with client.stream(
        "POST",
        "/v1/responses",
        json={
            "input": "Use calculator please",
            "stream": True,
            "tools": [{"type": "function", "name": "calculator", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}],
            "tool_choice": "required",
            "stream_options": {"include_obfuscation": False},
            "store": True,
        },
    ) as response:
        text = "".join(response.iter_text())

    events = parse_sse_events(text)
    event_types = [event["event"] for event in events]
    assert "response.output_item.added" in event_types
    assert "response.function_call_arguments.delta" in event_types
    assert "response.function_call_arguments.done" in event_types
    assert event_types[-1] == "response.completed"

    added = next(event for event in events if event["event"] == "response.output_item.added" and event["data"]["item"]["type"] == "function_call")
    done = next(event for event in events if event["event"] == "response.function_call_arguments.done")
    deltas = [event["data"]["delta"] for event in events if event["event"] == "response.function_call_arguments.delta"]
    assert added["data"]["item"]["arguments"] == ""
    assert "".join(deltas) == done["data"]["arguments"] == '{"expression":"string"}'

    terminal_response = events[-1]["data"]["response"]
    tool_item = terminal_response["output"][0]
    assert tool_item["type"] == "function_call"
    assert tool_item["arguments"] == done["data"]["arguments"]

    retrieved = client.get(f"/v1/responses/{terminal_response['id']}").json()
    assert retrieved["output"] == terminal_response["output"]


def test_streaming_web_search_output_item_events(web_search_client):
    with web_search_client.stream(
        "POST",
        "/v1/responses",
        json={
            "input": "Search the web for latest stream search details",
            "tools": [{"type": "web_search"}],
            "metadata": mock_metadata(tool_call="web_search"),
            "stream": True,
            "store": True,
        },
    ) as response:
        text = "".join(response.iter_text())

    events = parse_sse_events(text)
    event_types = [event["event"] for event in events]
    assert event_types[-1] == "response.completed"
    web_added = next(event for event in events if event["event"] == "response.output_item.added" and event["data"]["item"]["type"] == "web_search_call")
    web_done = next(event for event in events if event["event"] == "response.output_item.done" and event["data"]["item"]["type"] == "web_search_call")
    message_added = next(event for event in events if event["event"] == "response.output_item.added" and event["data"]["item"]["type"] == "message")
    assert events.index(web_added) < events.index(web_done) < events.index(message_added)
    assert "sources" not in web_added["data"]["item"]["action"]

    terminal_response = events[-1]["data"]["response"]
    assert terminal_response["output"][0]["type"] == "web_search_call"
    assert terminal_response["output"][1]["type"] == "message"
    assert terminal_response["output"][1]["content"][0]["annotations"][0]["type"] == "url_citation"

    retrieved = web_search_client.get(f"/v1/responses/{terminal_response['id']}").json()
    assert retrieved["output"] == terminal_response["output"]


def test_streaming_web_search_failure_events(web_search_client):
    with web_search_client.stream(
        "POST",
        "/v1/responses",
        json={
            "input": "Streaming web search failure fixture",
            "tools": [{"type": "web_search"}],
            "tool_choice": "required",
            "metadata": mock_metadata(web_search_error="timeout"),
            "stream": True,
        },
    ) as response:
        text = "".join(response.iter_text())

    events = parse_sse_events(text)
    event_types = [event["event"] for event in events]
    assert event_types[-2:] == ["response.failed", "error"]
    assert "response.output_item.added" not in event_types
    assert events[-1]["data"]["error"]["code"] == "web_search_timeout"


def test_streaming_image_generation_output_item_events(image_generation_client):
    with image_generation_client.stream(
        "POST",
        "/v1/responses",
        json={
            "input": "Generate image of a tiny stream house",
            "tools": [{"type": "image_generation", "quality": "low"}],
            "metadata": mock_metadata(tool_call="image_generation"),
            "stream": True,
            "store": True,
        },
    ) as response:
        text = "".join(response.iter_text())

    events = parse_sse_events(text)
    event_types = [event["event"] for event in events]
    assert event_types[-1] == "response.completed"
    image_added = next(event for event in events if event["event"] == "response.output_item.added" and event["data"]["item"]["type"] == "image_generation_call")
    image_done = next(event for event in events if event["event"] == "response.output_item.done" and event["data"]["item"]["type"] == "image_generation_call")
    assert events.index(image_added) < events.index(image_done)
    assert image_done["data"]["item"]["result"]

    terminal_response = events[-1]["data"]["response"]
    assert terminal_response["output"][0]["type"] == "image_generation_call"
    assert terminal_response["output_text"] == ""

    retrieved = image_generation_client.get(f"/v1/responses/{terminal_response['id']}").json()
    assert retrieved["output"] == terminal_response["output"]


def test_streaming_image_generation_failure_events(image_generation_client):
    with image_generation_client.stream(
        "POST",
        "/v1/responses",
        json={
            "input": "Streaming image generation failure fixture",
            "tools": [{"type": "image_generation"}],
            "tool_choice": {"type": "image_generation"},
            "metadata": mock_metadata(image_generation_error="timeout"),
            "stream": True,
        },
    ) as response:
        text = "".join(response.iter_text())

    events = parse_sse_events(text)
    event_types = [event["event"] for event in events]
    assert event_types[-2:] == ["response.failed", "error"]
    assert "response.output_item.added" not in event_types
    assert events[-1]["data"]["error"]["code"] == "image_generation_timeout"


def test_chat_completions_streaming(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gpt-oss-120b",
            "messages": [{"role": "user", "content": "chat stream"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        text = "".join(response.iter_text())

    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in text

    payloads = [
        json.loads(line.removeprefix("data:").strip())
        for line in text.splitlines()
        if line.startswith("data: {")
    ]
    streamed_text = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
    )

    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert streamed_text == "Mock response: chat stream "
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-1]["usage"]["total_tokens"] > 0
