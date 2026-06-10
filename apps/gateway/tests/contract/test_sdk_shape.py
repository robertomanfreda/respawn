import socket
import threading
import time

import httpx
import pytest
from openai import NotFoundError, OpenAI
from uvicorn import Config, Server

from src.adapters.mock_control import mock_metadata
from src.config import get_settings
from src.main import create_app


def test_openai_sdk_compatible_shape(client):
    response = client.post("/v1/responses", json={"model": "gpt-oss-120b", "input": "sdk shape"})
    body = response.json()
    assert body["object"] == "response"
    assert body["output"][0]["type"] == "message"
    assert body["output_text"] == "Mock response: sdk shape"
    assert body["usage"]["total_tokens"] >= 0
    assert body["usage"]["input_tokens_details"]["cached_tokens"] == 0
    assert body["usage"]["output_tokens_details"]["reasoning_tokens"] == 0


def test_official_openai_python_sdk_create(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'sdk.db'}")
    monkeypatch.setenv("MODEL_BACKEND", "mock")
    monkeypatch.setenv("AUTH_DISABLED", "true")
    get_settings.cache_clear()

    port = _free_port()
    server = Server(Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_server(port)

    try:
        sdk = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="local-dev-key")
        response = sdk.responses.create(model="gpt-oss-120b", input="sdk create", text={"format": {"type": "text"}}, metadata={"sdk": "phase-1"})
        assert response.id.startswith("resp_")
        assert response.output_text == "Mock response: sdk create"
        body = response.model_dump()
        assert body["store"] is True
        assert body["text"]["format"]["type"] == "text"
        assert body["metadata"] == {"sdk": "phase-1"}
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        get_settings.cache_clear()


def test_official_openai_python_sdk_stream(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'sdk_stream.db'}")
    monkeypatch.setenv("MODEL_BACKEND", "mock")
    monkeypatch.setenv("AUTH_DISABLED", "true")
    get_settings.cache_clear()

    port = _free_port()
    server = Server(Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_server(port)

    try:
        sdk = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="local-dev-key")
        with sdk.responses.stream(model="gpt-oss-120b", input="sdk stream") as stream:
            events = list(stream)
        event_types = [event.type for event in events]
        assert "response.output_text.delta" in event_types
        assert "response.completed" in event_types
        assert [getattr(event, "sequence_number", None) for event in events] == list(range(len(events)))
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        get_settings.cache_clear()


def test_official_openai_python_sdk_retrieve_and_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'sdk_state.db'}")
    monkeypatch.setenv("MODEL_BACKEND", "mock")
    monkeypatch.setenv("AUTH_DISABLED", "true")
    get_settings.cache_clear()

    port = _free_port()
    server = Server(Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_server(port)

    try:
        sdk = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="local-dev-key")
        created = sdk.responses.create(model="gpt-oss-120b", input="sdk state", metadata={"roundtrip": "true"}, temperature=0)
        retrieved = sdk.responses.retrieve(created.id)
        deleted = sdk.responses.delete(created.id)
        assert retrieved.id == created.id
        retrieved_body = retrieved.model_dump()
        assert retrieved_body["metadata"] == {"roundtrip": "true"}
        assert retrieved_body["temperature"] == 0
        assert deleted is None
        with pytest.raises(NotFoundError):
            sdk.responses.retrieve(created.id)
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        get_settings.cache_clear()


def test_official_openai_python_sdk_background_cancel(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'sdk_background.db'}")
    monkeypatch.setenv("MODEL_BACKEND", "mock")
    monkeypatch.setenv("AUTH_DISABLED", "true")
    get_settings.cache_clear()

    port = _free_port()
    server = Server(Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_server(port)

    try:
        sdk = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="local-dev-key")
        created = sdk.responses.create(model="gpt-oss-120b", input="background sdk", background=True, metadata=mock_metadata(delay_seconds=0.15))
        cancelled = sdk.responses.cancel(created.id)
        assert created.background is True
        assert cancelled.id == created.id
        assert cancelled.status == "cancelled"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        get_settings.cache_clear()


def test_official_openai_python_sdk_function_call_followup(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'sdk_tools.db'}")
    monkeypatch.setenv("MODEL_BACKEND", "mock")
    monkeypatch.setenv("AUTH_DISABLED", "true")
    get_settings.cache_clear()

    port = _free_port()
    server = Server(Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_server(port)

    try:
        sdk = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="local-dev-key")
        tools = [
            {
                "type": "function",
                "name": "calculator",
                "description": "Evaluate a small arithmetic expression.",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
            }
        ]
        first = sdk.responses.create(model="gpt-oss-120b", input="Use calculator please", tools=tools, tool_choice="required")
        first_body = first.model_dump()
        call = first_body["output"][0]
        assert call["type"] == "function_call"
        assert call["name"] == "calculator"
        assert call["arguments"] == '{"expression":"string"}'

        second = sdk.responses.create(
            model="gpt-oss-120b",
            previous_response_id=first.id,
            input=[{"type": "function_call_output", "call_id": call["call_id"], "output": "4"}],
            tools=tools,
        )
        assert second.output_text == "Tool result: 4"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        get_settings.cache_clear()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_server(port: int) -> None:
    deadline = time.monotonic() + 5
    url = f"http://127.0.0.1:{port}/healthz"
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=0.5).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.05)
    raise AssertionError("Uvicorn test server did not start.")
