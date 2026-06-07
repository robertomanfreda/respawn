import base64
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.config import get_settings
from src.main import create_app


TERMINAL_STATUSES = {"completed", "failed", "cancelled", "incomplete"}
TINY_RED_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP8z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
TINY_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
    b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    b"5 0 obj\n<< /Length 68 >>\nstream\n"
    b"BT /F1 12 Tf 36 100 Td (Respawn PDF marker word: quartz.) Tj ET\n"
    b"endstream\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF"
)


def _data_url(mime_type: str, payload_base64: str) -> str:
    return f"data:{mime_type};base64,{payload_base64}"


def test_list_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [{"id": "gpt-oss-120b", "object": "model", "created": 0, "owned_by": "mock"}],
    }


def test_list_models_root_alias(client):
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "gpt-oss-120b"


def test_basic_response_create_and_retrieve(client):
    created = client.post("/v1/responses", json={"model": "gpt-oss-120b", "input": "hello"}).json()
    assert created["id"].startswith("resp_")
    assert created["output"][0]["content"][0]["text"] == "Mock response: hello"
    assert created["output"][0]["content"][0]["annotations"] == []
    assert created["output"][0]["content"][0]["logprobs"] == []
    assert created["output_text"] == "Mock response: hello"
    assert created["parallel_tool_calls"] is False
    assert created["previous_response_id"] is None
    assert created["service_tier"] == "auto"
    assert created["store"] is True
    assert created["temperature"] == 1
    assert created["text"] == {"format": {"type": "text"}}
    assert created["tool_choice"] == "auto"
    assert created["top_p"] == 1
    assert created["truncation"] == "disabled"
    assert created["usage"]["input_tokens_details"]["cached_tokens"] == 0
    assert created["usage"]["output_tokens_details"]["reasoning_tokens"] == 0

    retrieved = client.get(f"/v1/responses/{created['id']}").json()
    assert retrieved["id"] == created["id"]
    assert retrieved["output_text"] == "Mock response: hello"
    assert retrieved["text"] == created["text"]
    assert retrieved["store"] is True


def test_response_request_settings_round_trip_through_retrieve(client):
    payload = {
        "model": "gpt-oss-120b",
        "input": "shape settings",
        "metadata": {"ticket": "phase-1"},
        "temperature": 0.2,
        "top_p": 0.9,
        "max_output_tokens": 16,
        "service_tier": "default",
        "text": {"format": {"type": "text"}},
        "safety_identifier": "safe-local-user",
        "store": True,
    }
    created = client.post("/v1/responses", json=payload).json()
    retrieved = client.get(f"/v1/responses/{created['id']}").json()

    for body in (created, retrieved):
        assert body["metadata"] == {"ticket": "phase-1"}
        assert body["temperature"] == 0.2
        assert body["top_p"] == 0.9
        assert body["max_output_tokens"] == 16
        assert body["parallel_tool_calls"] is False
        assert body["service_tier"] == "default"
        assert body["text"] == {"format": {"type": "text"}}
        assert body["safety_identifier"] == "safe-local-user"
        assert body["store"] is True


def test_max_output_token_exhaustion_marks_response_incomplete(client):
    created = client.post("/v1/responses", json={"input": "please produce more than one token", "max_output_tokens": 1}).json()

    assert created["status"] == "incomplete"
    assert created["incomplete_details"] == {"reason": "max_tokens"}
    assert created["output_text"] == "Mock"

    retrieved = client.get(f"/v1/responses/{created['id']}").json()
    assert retrieved["status"] == "incomplete"
    assert retrieved["incomplete_details"] == {"reason": "max_tokens"}


def test_response_input_items_list(client):
    created = client.post("/v1/responses", json={"input": "list my input", "store": True}).json()

    response = client.get(f"/v1/responses/{created['id']}/input_items")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["has_more"] is False
    assert body["first_id"] == body["data"][0]["id"]
    assert body["data"][0]["type"] == "message"
    assert body["data"][0]["role"] == "user"
    assert body["data"][0]["content"] == [{"type": "input_text", "text": "list my input"}]


def test_response_input_items_are_stored_with_stable_ids_and_pagination(client):
    created = client.post(
        "/v1/responses",
        json={
            "input": [
                {"role": "user", "content": "first stored input"},
                {"role": "user", "content": "second stored input"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "stored reasoning"}]},
            ],
            "store": True,
        },
    ).json()

    first_read = client.get(f"/v1/responses/{created['id']}/input_items?order=asc&limit=2").json()
    second_read = client.get(f"/v1/responses/{created['id']}/input_items?order=asc&limit=2").json()

    assert [item["id"] for item in first_read["data"]] == [item["id"] for item in second_read["data"]]
    assert first_read["has_more"] is True
    assert first_read["data"][0]["content"] == [{"type": "input_text", "text": "first stored input"}]
    assert first_read["data"][1]["content"] == [{"type": "input_text", "text": "second stored input"}]

    after_first = client.get(f"/v1/responses/{created['id']}/input_items?order=asc&after={first_read['data'][0]['id']}&limit=1").json()
    assert after_first["data"][0]["id"] == first_read["data"][1]["id"]

    before_second = client.get(f"/v1/responses/{created['id']}/input_items?order=asc&before={first_read['data'][1]['id']}&limit=10").json()
    assert [item["id"] for item in before_second["data"]] == [first_read["data"][0]["id"]]


def test_store_false_response_input_items_are_not_exposed(client):
    created = client.post("/v1/responses", json={"input": "hidden input", "store": False}).json()

    response = client.get(f"/v1/responses/{created['id']}/input_items")

    assert response.status_code == 404


def test_retrieve_uses_canonical_output_items(client):
    created = client.post("/v1/responses", json={"input": "canonical output", "store": True}).json()
    database_url = client.app.state.settings.database_url
    database_path = database_url.removeprefix("sqlite+aiosqlite:///")
    stale_output = [
        {
            "id": "msg_stale",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "stale response json", "annotations": [], "logprobs": []}],
        }
    ]
    with sqlite3.connect(database_path) as connection:
        connection.execute("update responses set output_json = ? where id = ?", (json.dumps(stale_output), created["id"]))
        connection.commit()

    retrieved = client.get(f"/v1/responses/{created['id']}").json()

    assert retrieved["output_text"] == "Mock response: canonical output"
    assert retrieved["output"][0]["id"] == created["output"][0]["id"]


def test_response_input_tokens_count(client):
    response = client.post("/v1/responses/input_tokens", json={"model": "gpt-oss-120b", "input": "count these tokens"})

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response.input_tokens"
    assert body["input_tokens"] > 0
    assert body["input_tokens_details"]["cached_tokens"] == 0


def test_prompt_cache_reports_cached_tokens(client):
    payload = {
        "model": "gpt-oss-120b",
        "input": "shared prefix token token token token token variable-a",
        "prompt_cache_key": "integration-cache",
        "prompt_cache_retention": "in_memory",
    }
    first = client.post("/v1/responses", json=payload).json()
    second = client.post("/v1/responses", json={**payload, "input": "shared prefix token token token token token variable-b"}).json()
    counted = client.post("/v1/responses/input_tokens", json={**payload, "input": "shared prefix token token token token token variable-c"}).json()

    assert first["usage"]["input_tokens_details"]["cached_tokens"] == 0
    assert second["usage"]["input_tokens_details"]["cached_tokens"] > 0
    assert counted["input_tokens_details"]["cached_tokens"] > 0


def test_reasoning_output_item_and_usage(client):
    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": "reason briefly",
            "reasoning": {"effort": "low", "summary": "auto"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output"][0]["type"] == "reasoning"
    assert body["output"][0]["summary"][0]["type"] == "summary_text"
    assert "Estimated reasoning tokens" in body["output"][0]["summary"][0]["text"]
    assert body["output"][1]["type"] == "message"
    assert body["output_text"] == "Mock response: reason briefly"
    assert body["usage"]["output_tokens_details"]["reasoning_tokens"] > 0


def test_reasoning_input_item_round_trips_without_becoming_user_text(client):
    response = client.post(
        "/v1/responses",
        json={
            "input": [
                {"id": "rs_input", "type": "reasoning", "summary": [{"type": "summary_text", "text": "prior reasoning"}]},
                {"role": "user", "content": "hello after reasoning"},
            ],
            "store": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output_text"] == "Mock response: hello after reasoning"
    input_items = client.get(f"/v1/responses/{body['id']}/input_items?order=asc").json()
    assert input_items["data"][0]["type"] == "reasoning"


def test_background_response_create_poll_complete(client):
    created = client.post("/v1/responses", json={"input": "background slow complete", "background": True, "store": True})

    assert created.status_code == 200
    body = created.json()
    assert body["background"] is True
    assert body["status"] in {"queued", "in_progress", "completed"}
    if body["status"] != "completed":
        assert body["output"] == []

    completed = poll_response(client, body["id"], expected={"completed"})

    assert completed["background"] is True
    assert completed["output_text"] == "Mock response: background slow complete"
    assert completed["usage"]["total_tokens"] > 0


def test_background_cancel_is_terminal_and_idempotent(client):
    created = client.post("/v1/responses", json={"input": "background slow cancel", "background": True, "store": True}).json()

    cancelled = client.post(f"/v1/responses/{created['id']}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    second = client.post(f"/v1/responses/{created['id']}/cancel")
    assert second.status_code == 200
    assert second.json()["status"] == "cancelled"

    time.sleep(0.25)
    retrieved = client.get(f"/v1/responses/{created['id']}").json()
    assert retrieved["status"] == "cancelled"


def test_background_store_false_is_explicitly_invalid(client):
    response = client.post("/v1/responses", json={"input": "hello", "background": True, "store": False})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["param"] == "store"


def test_streaming_background_response_is_explicitly_unsupported(client):
    response = client.post("/v1/responses", json={"input": "hello", "background": True, "stream": True})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == "stream"


def test_stream_options_are_validated(client):
    not_streaming = client.post("/v1/responses", json={"input": "hello", "stream_options": {"include_obfuscation": False}})
    assert not_streaming.status_code == 400
    assert not_streaming.json()["error"]["param"] == "stream_options"

    unknown_option = client.post("/v1/responses", json={"input": "hello", "stream": True, "stream_options": {"unknown": True}})
    assert unknown_option.status_code == 400
    assert unknown_option.json()["error"]["code"] == "unsupported_parameter"
    assert unknown_option.json()["error"]["param"] == "stream_options.unknown"

    invalid_obfuscation = client.post("/v1/responses", json={"input": "hello", "stream": True, "stream_options": {"include_obfuscation": "no"}})
    assert invalid_obfuscation.status_code == 400
    assert invalid_obfuscation.json()["error"]["param"] == "stream_options.include_obfuscation"


def test_cancel_non_background_response_is_invalid(client):
    created = client.post("/v1/responses", json={"input": "not background"}).json()

    response = client.post(f"/v1/responses/{created['id']}/cancel")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_background_timeout_marks_response_failed(tmp_path, monkeypatch):
    with configured_client(
        tmp_path,
        monkeypatch,
        DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'timeout.db'}",
        BACKGROUND_JOB_TIMEOUT_SECONDS="0.05",
    ) as client:
        created = client.post("/v1/responses", json={"input": "background timeout", "background": True, "store": True}).json()
        failed = poll_response(client, created["id"], expected={"failed"})

    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "background_timeout"


def test_background_metrics_include_job_signals(client):
    created = client.post("/v1/responses", json={"input": "background slow metrics", "background": True, "store": True}).json()
    poll_response(client, created["id"], expected={"completed"})

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "gateway_background_jobs_total" in response.text
    assert "gateway_background_job_latency_seconds_bucket" in response.text
    assert "gateway_background_jobs_running" in response.text


def test_conversation_field_is_out_of_scope(client):
    response = client.post("/v1/responses", json={"input": "hello", "conversation": "conv_123"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_phase_one_request_validation_is_explicit(client):
    cases = [
        ({"input": "hello", "service_tier": "vip"}, "service_tier"),
        ({"input": "hello", "truncation": "auto"}, "truncation"),
        ({"input": "hello", "top_logprobs": 1}, "top_logprobs"),
        ({"input": "hello", "metadata": {"not_string": 1}}, "metadata.not_string"),
    ]

    for payload, param in cases:
        response = client.post("/v1/responses", json=payload)
        assert response.status_code == 400
        assert response.json()["error"]["param"] == param


def test_image_input_with_text_model_returns_capability_error(client):
    response = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is this?"},
                        {"type": "input_image", "image_url": "https://example.com/image.png"},
                    ],
                }
            ]
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_model_capability"
    assert response.json()["error"]["param"] == "model"


def test_image_input_with_vision_model_is_normalized_and_stored(client):
    response = client.post(
        "/v1/responses",
        json={
            "model": "moondream:latest",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What color is the square?"},
                        {"type": "input_image", "image_url": _data_url("image/png", TINY_RED_PNG_BASE64)},
                    ],
                }
            ],
            "store": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "red" in body["output_text"].lower()
    input_items = client.get(f"/v1/responses/{body['id']}/input_items?order=asc").json()["data"]
    image_part = input_items[0]["content"][1]
    assert image_part["type"] == "input_image"
    assert image_part["detail"] == "auto"
    assert image_part["mime_type"] == "image/png"
    assert image_part["image_base64"] == TINY_RED_PNG_BASE64


def test_text_file_input_is_extracted_stored_and_replayed(client):
    file_text = "Respawn file marker word: cobalt."
    created = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "filename": "facts.txt", "file_data": _data_url("text/plain", base64.b64encode(file_text.encode()).decode())},
                        {"type": "input_text", "text": "Repeat the marker word."},
                    ],
                }
            ],
            "store": True,
        },
    ).json()

    assert "cobalt" in created["output_text"]
    input_items = client.get(f"/v1/responses/{created['id']}/input_items?order=asc").json()["data"]
    file_part = input_items[0]["content"][0]
    assert file_part["type"] == "input_file"
    assert file_part["filename"] == "facts.txt"
    assert file_part["text"] == file_text

    followup = client.post("/v1/responses", json={"previous_response_id": created["id"], "input": "Use the previous file marker word."}).json()
    assert "cobalt" in followup["output_text"]


def test_csv_and_pdf_file_inputs_are_extracted(client):
    csv_text = "item,count\nalpha,2\nbeta,5\n"
    csv_response = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "filename": "table.csv", "file_data": _data_url("text/csv", base64.b64encode(csv_text.encode()).decode())},
                        {"type": "input_text", "text": "Which row has count 5?"},
                    ],
                }
            ]
        },
    ).json()
    assert "beta" in csv_response["output_text"]

    pdf_response = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "filename": "tiny.pdf", "file_data": _data_url("application/pdf", base64.b64encode(TINY_PDF_BYTES).decode())},
                        {"type": "input_text", "text": "What is the PDF marker word?"},
                    ],
                }
            ]
        },
    ).json()
    assert "quartz" in pdf_response["output_text"]


def test_file_id_and_audio_inputs_are_explicitly_unsupported(client):
    file_id_response = client.post(
        "/v1/responses",
        json={"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "file_123"}]}]},
    )
    assert file_id_response.status_code == 400
    assert file_id_response.json()["error"]["code"] == "unsupported_parameter"
    assert file_id_response.json()["error"]["param"] == "input.0.content.0.file_id"

    audio_response = client.post(
        "/v1/responses",
        json={"input": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}]}]},
    )
    assert audio_response.status_code == 400
    assert audio_response.json()["error"]["code"] == "unsupported_parameter"
    assert audio_response.json()["error"]["param"] == "input.0.content.0.type"


def test_builtin_tools_are_explicitly_unsupported(client):
    response = client.post("/v1/responses", json={"input": "search", "tools": [{"type": "web_search"}]})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == "tools.0.type"


def test_invalid_prompt_cache_retention_is_explicit(client):
    response = client.post("/v1/responses", json={"input": "hello", "prompt_cache_retention": "forever"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == "prompt_cache_retention"


def test_metrics_include_model_gateway_signals(client):
    client.post("/v1/responses", json={"model": "gpt-oss-120b", "input": "metrics"})
    first = client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": "Use calculator please",
            "tools": [{"type": "function", "name": "calculator", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}],
            "tool_choice": "required",
        },
    ).json()
    client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "previous_response_id": first["id"],
            "input": [{"type": "function_call_output", "call_id": first["output"][0]["call_id"], "output": "4"}],
            "tools": [{"type": "function", "name": "calculator", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}],
        },
    )

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "gateway_responses_total" in response.text
    assert "gateway_response_latency_seconds_bucket" in response.text
    assert "gateway_inflight_responses" in response.text
    assert "gateway_model_token_usage_total" in response.text
    assert "gateway_function_tool_requests_total" in response.text
    assert "gateway_function_tool_calls_total" in response.text
    assert "gateway_function_tool_outputs_total" in response.text


def test_stateful_previous_response_id(client):
    first = client.post("/v1/responses", json={"input": "Mi chiamo Roberto.", "store": True}).json()
    second = client.post("/v1/responses", json={"previous_response_id": first["id"], "input": "Come mi chiamo?", "store": True}).json()
    assert second["status"] == "completed"


def test_delete_then_get_and_previous_response_fail(client):
    created = client.post("/v1/responses", json={"input": "delete me"}).json()
    deleted = client.delete(f"/v1/responses/{created['id']}").json()
    assert deleted == {"id": created["id"], "object": "response.deleted", "deleted": True}
    assert client.get(f"/v1/responses/{created['id']}").status_code == 404
    response = client.post("/v1/responses", json={"previous_response_id": created["id"], "input": "again"})
    assert response.status_code == 404


def test_function_tool_request_emits_function_call_without_executing(client):
    response = client.post(
        "/v1/responses",
        json={
            "input": "Use calculator please",
            "tools": [{"type": "function", "name": "calculator", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["output_text"] == ""
    assert body["parallel_tool_calls"] is True
    assert body["output"] == [
        {
            "id": body["output"][0]["id"],
            "type": "function_call",
            "status": "completed",
            "call_id": "call_mock_calculator",
            "name": "calculator",
            "arguments": '{"expression":"2+2"}',
        }
    ]


def test_function_call_output_followup_with_previous_response_id(client):
    first = client.post(
        "/v1/responses",
        json={
            "input": "Use calculator please",
            "tools": [{"type": "function", "name": "calculator", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}],
            "tool_choice": {"type": "function", "name": "calculator"},
            "store": True,
        },
    ).json()
    call = first["output"][0]

    second = client.post(
        "/v1/responses",
        json={
            "previous_response_id": first["id"],
            "input": [{"type": "function_call_output", "call_id": call["call_id"], "output": "4"}],
            "tools": [{"type": "function", "name": "calculator", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}],
            "store": True,
        },
    )

    assert second.status_code == 200
    body = second.json()
    assert body["output"][0]["type"] == "message"
    assert body["output_text"] == "Tool result: 4"

    input_items = client.get(f"/v1/responses/{body['id']}/input_items?order=asc").json()
    assert input_items["data"][0]["type"] == "function_call_output"
    assert input_items["data"][0]["call_id"] == call["call_id"]


def test_manual_function_call_and_output_followup(client):
    response = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_manual",
                    "name": "calculator",
                    "arguments": "{\"expression\":\"6*7\"}",
                },
                {"type": "function_call_output", "call_id": "call_manual", "output": "{\"result\":42}"},
            ],
            "tools": [{"type": "function", "name": "calculator", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}],
        },
    )

    assert response.status_code == 200
    assert response.json()["output_text"] == 'Tool result: {"result":42}'


def test_manual_followup_can_resend_stored_function_call_item(client):
    first = client.post(
        "/v1/responses",
        json={
            "input": "Use calculator please",
            "tools": [{"type": "function", "name": "calculator", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}],
            "tool_choice": "required",
            "store": True,
        },
    ).json()
    call = first["output"][0]

    second = client.post(
        "/v1/responses",
        json={
            "input": [call, {"type": "function_call_output", "call_id": call["call_id"], "output": "4"}],
            "tools": [{"type": "function", "name": "calculator", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}],
            "store": True,
        },
    )

    assert second.status_code == 200
    body = second.json()
    assert body["output_text"] == "Tool result: 4"
    input_items = client.get(f"/v1/responses/{body['id']}/input_items?order=asc").json()["data"]
    stored_call = next(item for item in input_items if item["type"] == "function_call")
    assert stored_call["call_id"] == call["call_id"]
    assert stored_call["id"] != call["id"]


def test_function_call_output_requires_matching_call(client):
    response = client.post("/v1/responses", json={"input": [{"type": "function_call_output", "call_id": "call_missing", "output": "4"}]})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_tool_call_output"
    assert response.json()["error"]["param"] == "input"


def test_legacy_tool_result_is_explicitly_unsupported(client):
    response = client.post("/v1/responses", json={"input": [{"type": "tool_result", "call_id": "call_1", "output": "4"}]})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == "input.0.type"


def test_structured_output_repairs_once(client):
    response = client.post("/v1/responses", json={"input": "hello", "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}})
    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == "{}"


def test_structured_output_accepts_sdk_text_format(client):
    response = client.post(
        "/v1/responses",
        json={
            "input": "hello",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "summary",
                    "schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
                }
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == '{"summary":"string"}'
    assert response.json()["text"]["format"]["type"] == "json_schema"


def test_structured_output_repair_failure(client):
    response = client.post("/v1/responses", json={"input": "repair failure", "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}})
    assert response.status_code == 502


def test_store_false_response_is_not_retrievable(client):
    created = client.post("/v1/responses", json={"input": "ephemeral", "store": False}).json()
    assert client.get(f"/v1/responses/{created['id']}").status_code == 404


def test_validation_errors_are_openai_shaped(client):
    response = client.post("/v1/responses", json={"input": {"not": "valid"}})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_function_tool_loop_is_client_driven_not_executed_by_respawn(client):
    response = client.post(
        "/v1/responses",
        json={
            "input": "loop forever",
            "tools": [{"type": "function", "name": "echo", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}],
            "tool_choice": "required",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["output"][0]["type"] == "function_call"
    assert body["output"][0]["name"] == "echo"
    assert body["output_text"] == ""


def poll_response(client, response_id: str, *, expected: set[str], timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    last_body = {}
    while time.monotonic() < deadline:
        response = client.get(f"/v1/responses/{response_id}")
        assert response.status_code == 200
        last_body = response.json()
        if last_body["status"] in expected:
            return last_body
        assert last_body["status"] not in TERMINAL_STATUSES, last_body
        time.sleep(0.02)
    raise AssertionError(f"response {response_id} did not reach {expected}; last body: {last_body}")


@contextmanager
def configured_client(tmp_path, monkeypatch: pytest.MonkeyPatch, **env: str) -> Iterator[TestClient]:
    defaults = {
        "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        "MODEL_BACKEND": "mock",
        "AUTH_DISABLED": "true",
        "DEFAULT_MODEL": "gpt-oss-120b",
        "PROMPT_CACHE_MIN_TOKENS": "8",
    }
    previous_env = {key: os.environ.get(key) for key in defaults | env}
    for key, value in {**defaults, **env}.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            yield client
    finally:
        for key, value in previous_env.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        get_settings.cache_clear()
