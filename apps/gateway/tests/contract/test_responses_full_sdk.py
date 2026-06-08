from __future__ import annotations

from contextlib import contextmanager
import os
import socket
import threading
import time
from typing import Iterator

import pytest
from openai import BadRequestError, ConflictError, InternalServerError, NotFoundError, OpenAI, UnprocessableEntityError
from uvicorn import Config, Server

from src.config import get_settings
from src.main import create_app


BASE_ENV = {
    "MODEL_BACKEND": "mock",
    "AUTH_DISABLED": "true",
    "DEFAULT_MODEL": "gpt-oss-120b",
    "FILE_STORAGE_BACKEND": "database",
    "PROMPT_CACHE_MIN_TOKENS": "8",
}

RESPONSE_SNAPSHOT_KEYS = {
    "id",
    "object",
    "created_at",
    "status",
    "model",
    "input",
    "output",
    "output_text",
    "usage",
    "metadata",
    "error",
    "incomplete_details",
    "text",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "truncation",
}


def test_python_sdk_responses_files_headers_and_pagination(tmp_path, monkeypatch):
    with sdk_server(tmp_path, monkeypatch) as sdk:
        raw = sdk.responses.with_raw_response.create(
            model="gpt-oss-120b",
            input=[
                {"role": "user", "content": "first SDK item"},
                {"role": "user", "content": "second SDK item"},
            ],
            metadata={"sdk": "phase-14"},
            text={"format": {"type": "text"}},
            extra_headers={"x-request-id": "req_phase14_sdk"},
        )
        created = raw.parse()
        assert raw.headers["x-request-id"] == "req_phase14_sdk"
        assert created._request_id == "req_phase14_sdk"
        assert RESPONSE_SNAPSHOT_KEYS.issubset(created.model_dump().keys())

        retrieved = sdk.responses.retrieve(created.id)
        assert retrieved.id == created.id
        assert retrieved.metadata == {"sdk": "phase-14"}

        first_page = sdk.responses.input_items.list(created.id, order="asc", limit=1)
        assert len(first_page.data) == 1
        first_page_body = first_page.model_dump()
        assert first_page_body["first_id"] == first_page.data[0].id
        assert first_page_body["last_id"] == first_page.data[-1].id
        assert first_page.has_more is True
        second_page = sdk.responses.input_items.list(created.id, order="asc", limit=1, after=first_page.data[0].id)
        assert second_page.data[0].id != first_page.data[0].id

        uploaded = sdk.files.create(file=("phase14.txt", b"phase 14 sdk file marker"), purpose="user_data")
        assert uploaded.id.startswith("file_")
        assert sdk.files.retrieve(uploaded.id).filename == "phase14.txt"
        assert "phase 14" in sdk.files.content(uploaded.id).text
        file_page = sdk.files.list(order="asc", limit=1)
        file_page_body = file_page.model_dump()
        assert file_page_body["first_id"] == uploaded.id
        assert file_page_body["last_id"] == uploaded.id
        assert file_page.has_more is False
        deleted_file = sdk.files.delete(uploaded.id)
        assert deleted_file.deleted is True

        assert sdk.responses.delete(created.id) is None
        with pytest.raises(NotFoundError):
            sdk.responses.retrieve(created.id)


def test_python_sdk_stream_background_and_function_followup(tmp_path, monkeypatch):
    with sdk_server(tmp_path, monkeypatch) as sdk:
        with sdk.responses.stream(model="gpt-oss-120b", input="sdk stream phase 14") as stream:
            events = list(stream)
        event_types = [event.type for event in events]
        assert "response.output_text.delta" in event_types
        assert event_types[-1] in {"response.completed", "response.incomplete"}
        assert [getattr(event, "sequence_number", None) for event in events] == list(range(len(events)))

        background = sdk.responses.create(model="gpt-oss-120b", input="background slow sdk", background=True)
        cancelled = sdk.responses.cancel(background.id)
        assert background.background is True
        assert cancelled.status in {"cancelled", "completed", "incomplete"}

        tools = [
            {
                "type": "function",
                "name": "calculator",
                "description": "Evaluate a small arithmetic expression.",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
            }
        ]
        first = sdk.responses.create(model="gpt-oss-120b", input="Use calculator please", tools=tools, tool_choice="required")
        call = first.model_dump()["output"][0]
        assert call["type"] == "function_call"
        followup = sdk.responses.create(
            model="gpt-oss-120b",
            previous_response_id=first.id,
            input=[{"type": "function_call_output", "call_id": call["call_id"], "output": "4"}],
            tools=tools,
        )
        assert followup.output_text == "Tool result: 4"


def test_python_sdk_idempotency_and_error_classes(tmp_path, monkeypatch):
    with sdk_server(tmp_path, monkeypatch) as sdk:
        headers = {"Idempotency-Key": "phase14-idempotency"}
        first = sdk.responses.create(model="gpt-oss-120b", input="idempotent body", extra_headers=headers)
        replayed = sdk.responses.create(model="gpt-oss-120b", input="idempotent body", extra_headers=headers)
        assert replayed.id == first.id
        with pytest.raises(ConflictError) as conflict:
            sdk.responses.create(model="gpt-oss-120b", input="different body", extra_headers=headers)
        assert conflict.value.status_code == 409
        assert conflict.value.response.headers["x-request-id"].startswith("req_")

        with pytest.raises(BadRequestError) as bad_request:
            sdk.responses.create(model="gpt-oss-120b", input="bad", user="legacy-user")
        assert bad_request.value.status_code == 400
        assert bad_request.value.body["code"] == "unsupported_parameter"

        with pytest.raises(NotFoundError) as missing:
            sdk.responses.retrieve("resp_missing")
        assert missing.value.status_code == 404
        assert missing.value.body["code"] == "not_found"

        with pytest.raises(UnprocessableEntityError) as validation:
            sdk.responses.create(model="gpt-oss-120b", input="invalid temperature", temperature=3)
        assert validation.value.status_code == 422
        assert validation.value.body["code"] == "validation_error"


def test_python_sdk_server_error_class(tmp_path, monkeypatch):
    with sdk_server(tmp_path, monkeypatch, FILE_STORAGE_BACKEND="broken") as sdk:
        with pytest.raises(InternalServerError) as server_error:
            sdk.files.create(file=("broken.txt", b"broken"), purpose="user_data")
        assert server_error.value.status_code == 500
        assert server_error.value.body["code"] == "invalid_file_storage_backend"
        assert server_error.value.response.headers["x-request-id"].startswith("req_")


@contextmanager
def sdk_server(tmp_path, monkeypatch, **env: str) -> Iterator[OpenAI]:
    keys = set(BASE_ENV) | {"DATABASE_URL", "FILE_STORAGE_BACKEND"} | set(env)
    previous = {key: os.environ.get(key) for key in keys}
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'sdk_contract.db'}")
    for key, value in BASE_ENV.items():
        monkeypatch.setenv(key, value)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()

    port = _free_port()
    server = Server(Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_server(port)
    try:
        yield OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="local-dev-key", max_retries=0)
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        for key, value in previous.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        get_settings.cache_clear()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_server(port: int) -> None:
    deadline = threading.Event()
    start = time.time()
    while time.time() - start < 5:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            deadline.wait(0.05)
    raise RuntimeError("SDK contract server did not start")
