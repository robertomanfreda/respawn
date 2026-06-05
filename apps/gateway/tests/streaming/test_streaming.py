import json


def test_streaming_event_lifecycle(client):
    with client.stream("POST", "/v1/responses", json={"input": "hello stream", "stream": True}) as response:
        text = "".join(response.iter_text())
    assert "event: response.created" in text
    assert "event: response.output_text.delta" in text
    assert "event: response.output_text.done" in text
    assert "event: response.content_part.done" in text
    assert "event: response.output_item.done" in text
    assert "event: response.completed" in text


def test_streaming_reasoning_events(client):
    with client.stream(
        "POST",
        "/v1/responses",
        json={"input": "hello reasoning stream", "stream": True, "reasoning": {"effort": "low", "summary": "auto"}},
    ) as response:
        text = "".join(response.iter_text())

    assert "event: response.reasoning_summary_text.done" in text
    assert '"type":"reasoning"' in text
    assert "Estimated reasoning tokens" in text
    assert "event: response.completed" in text


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
