import json

import httpx
import pytest
from prometheus_client import generate_latest

from src.adapters.ollama_backend import OllamaBackend
from src.schemas.errors import OpenAIError


@pytest.mark.asyncio
async def test_ollama_list_models_uses_models_endpoint():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "gpt-oss:120b", "object": "model", "created": 123, "owned_by": "ollama"}]},
        )

    adapter = OllamaBackend("http://ollama.test/v1", 10, transport=httpx.MockTransport(handler))
    result = await adapter.list_models()

    assert result.model_dump() == {
        "object": "list",
        "data": [{"id": "gpt-oss:120b", "object": "model", "created": 123, "owned_by": "ollama"}],
    }
    metrics = generate_latest().decode()
    assert 'gateway_backend_model_info{backend="ollama",model="gpt-oss:120b"} 1.0' in metrics


@pytest.mark.asyncio
async def test_ollama_non_streaming_payload_and_usage_mapping():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-oss:120b"
        assert payload["stream"] is False
        assert payload["think"] is False
        assert payload["options"]["num_predict"] == 8
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "ciao"},
                "prompt_eval_count": 3,
                "prompt_eval_duration": 1000000000,
                "eval_count": 2,
                "eval_duration": 500000000,
            },
        )

    adapter = OllamaBackend("http://ollama.test/v1", 10, transport=httpx.MockTransport(handler))
    result = await adapter.create_chat_completion({"model": "gpt-oss:120b", "messages": [{"role": "user", "content": "ciao"}], "max_tokens": 8})

    assert result.content == "ciao"
    assert result.usage == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


@pytest.mark.asyncio
async def test_ollama_reasoning_maps_to_think_and_usage_details():
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["think"] == "low"
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "thinking": "check the prompt carefully", "content": "ciao"},
                "prompt_eval_count": 3,
                "eval_count": 7,
            },
        )

    adapter = OllamaBackend("http://ollama.test/v1", 10, transport=httpx.MockTransport(handler))
    result = await adapter.create_chat_completion({"model": "gpt-oss:120b", "messages": [{"role": "user", "content": "ciao"}], "reasoning": {"effort": "low"}})

    assert result.content == "ciao"
    assert result.reasoning == "check the prompt carefully"
    assert result.usage["output_tokens_details"]["reasoning_tokens"] > 0


@pytest.mark.asyncio
async def test_ollama_reasoning_xhigh_maps_to_high_think_level():
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["think"] == "high"
        return httpx.Response(200, json={"message": {"role": "assistant", "thinking": "deep check", "content": "ok"}, "prompt_eval_count": 1, "eval_count": 2})

    adapter = OllamaBackend("http://ollama.test/v1", 10, transport=httpx.MockTransport(handler))
    result = await adapter.create_chat_completion({"model": "gpt-oss:120b", "messages": [], "reasoning": {"effort": "xhigh"}})

    assert result.content == "ok"
    assert result.reasoning == "deep check"


@pytest.mark.asyncio
async def test_ollama_undeclared_tool_calls_are_returned_as_text():
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "container.exec",
                                "arguments": {"action": "tool", "tool": "mcp.neuromem.neuromem.remember", "args": {}},
                            },
                        }
                    ],
                },
                "prompt_eval_count": 3,
                "eval_count": 2,
            },
        )

    adapter = OllamaBackend("http://ollama.test/v1", 10, transport=httpx.MockTransport(handler))
    result = await adapter.create_chat_completion({"model": "gpt-oss:120b", "messages": []})

    assert result.content == '{"action":"tool","tool":"mcp.neuromem.neuromem.remember","args":{}}'
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_ollama_tool_call_messages_are_mapped_to_native_arguments():
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        tool_call = payload["messages"][1]["tool_calls"][0]
        assert tool_call["function"]["arguments"] == {"expression": "2+2"}
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "4"},
                "prompt_eval_count": 3,
                "eval_count": 1,
            },
        )

    adapter = OllamaBackend("http://ollama.test/v1", 10, transport=httpx.MockTransport(handler))
    result = await adapter.create_chat_completion(
        {
            "model": "gpt-oss:120b",
            "messages": [
                {"role": "user", "content": "Use calculator."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": '{"expression":"2+2"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_123", "content": "4"},
            ],
        }
    )

    assert result.content == "4"
    assert result.usage == {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}


@pytest.mark.asyncio
async def test_ollama_multimodal_message_maps_images_and_file_text():
    image_base64 = "iVBORw0KGgo="

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        message = payload["messages"][0]
        assert message["images"] == [image_base64]
        assert "[Image input]" not in message["content"]
        assert "describe" in message["content"]
        assert "[File: facts.txt]" in message["content"]
        assert "marker word is cobalt" in message["content"]
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}, "prompt_eval_count": 3, "eval_count": 1})

    adapter = OllamaBackend("http://ollama.test/v1", 10, transport=httpx.MockTransport(handler))
    result = await adapter.create_chat_completion(
        {
            "model": "moondream:latest",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe"},
                        {"type": "input_image", "image_base64": image_base64, "mime_type": "image/png"},
                        {"type": "input_file", "filename": "facts.txt", "text": "marker word is cobalt"},
                    ],
                }
            ],
        }
    )

    assert result.content == "ok"


@pytest.mark.asyncio
async def test_ollama_streaming_maps_sse_chunks_and_usage():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"delta":{"content":"ci"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"ao"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    adapter = OllamaBackend("http://ollama.test/v1", 10, transport=httpx.MockTransport(handler))
    chunks = [chunk async for chunk in adapter.create_chat_completion_stream({"model": "gpt-oss:120b", "messages": []})]

    assert chunks == [
        {"type": "delta", "delta": "ci"},
        {"type": "delta", "delta": "ao"},
        {"type": "done", "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
    ]


@pytest.mark.asyncio
async def test_ollama_streaming_maps_jsonl_chunks_and_native_usage():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200,
            text=(
                '{"message":{"content":"ci"},"done":false}\n'
                '{"message":{"content":"ao"},"done":false}\n'
                '{"done":true,"prompt_eval_count":10,"prompt_eval_duration":1000000000,"eval_count":20,"eval_duration":2000000000}\n'
            ),
        )

    adapter = OllamaBackend("http://ollama.test/v1", 10, transport=httpx.MockTransport(handler))
    chunks = [chunk async for chunk in adapter.create_chat_completion_stream({"model": "gpt-oss:120b", "messages": []})]

    assert chunks == [
        {"type": "delta", "delta": "ci"},
        {"type": "delta", "delta": "ao"},
        {"type": "done", "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}},
    ]
    metrics = generate_latest().decode()
    assert 'gateway_backend_eval_tokens_total{backend="ollama",model="gpt-oss:120b",operation="chat_completion_stream",phase="prefill"}' in metrics
    assert 'gateway_backend_eval_duration_seconds_total{backend="ollama",model="gpt-oss:120b",operation="chat_completion_stream",phase="decode"}' in metrics
    assert 'gateway_backend_eval_tokens_per_second{backend="ollama",model="gpt-oss:120b",operation="chat_completion_stream",phase="decode"} 10.0' in metrics
    assert 'gateway_ollama_eval_tokens_total{model="gpt-oss:120b",operation="chat_completion_stream",phase="prefill"}' in metrics
    assert 'gateway_ollama_eval_duration_seconds_total{model="gpt-oss:120b",operation="chat_completion_stream",phase="decode"}' in metrics
    assert 'gateway_ollama_eval_tokens_per_second{model="gpt-oss:120b",operation="chat_completion_stream",phase="decode"} 10.0' in metrics


@pytest.mark.asyncio
async def test_ollama_streaming_maps_reasoning_chunks():
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["think"] == "medium"
        return httpx.Response(
            200,
            text=(
                '{"message":{"thinking":"thinking "},"done":false}\n'
                '{"message":{"content":"answer"},"done":false}\n'
                '{"done":true,"prompt_eval_count":10,"eval_count":20}\n'
            ),
        )

    adapter = OllamaBackend("http://ollama.test/v1", 10, transport=httpx.MockTransport(handler))
    chunks = [chunk async for chunk in adapter.create_chat_completion_stream({"model": "gpt-oss:120b", "messages": [], "reasoning": {"summary": "auto"}})]

    assert chunks == [
        {"type": "reasoning_delta", "delta": "thinking "},
        {"type": "delta", "delta": "answer"},
        {"type": "done", "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}},
    ]


@pytest.mark.asyncio
async def test_ollama_backend_error_maps_to_openai_error():
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    adapter = OllamaBackend("http://ollama.test/v1", 10, transport=httpx.MockTransport(handler))

    with pytest.raises(OpenAIError) as exc:
        await adapter.create_chat_completion({"model": "gpt-oss:120b", "messages": []})

    assert exc.value.status_code == 502
    assert exc.value.code == "backend_error"
