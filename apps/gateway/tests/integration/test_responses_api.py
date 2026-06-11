import base64
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.adapters.mock_control import mock_metadata
from src.config import get_settings
from src.main import create_app
from src.observability.logging import TRACE_LEVEL
from src.services.context_management import unseal_compaction_content
from src.services.reasoning_encryption import unseal_reasoning_content
from tests.helpers import backend_tool_names


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


def _upload_text_file(client, *, filename: str = "facts.txt", text: str = "Respawn file marker word: cobalt.", headers: dict[str, str] | None = None):
    return client.post(
        "/v1/files",
        headers=headers or {},
        data={"purpose": "user_data"},
        files={"file": (filename, text.encode(), "text/plain")},
    )


def test_list_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0] == {
        "id": "gpt-oss-120b",
        "object": "model",
        "created": 0,
        "owned_by": "mock",
        "context_window": 131072,
        "max_context_window": 131072,
        "effective_context_window_percent": 95,
    }


def test_list_models_root_alias(client):
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "gpt-oss-120b"
    assert response.json()["data"][0]["context_window"] == 131072


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


def test_request_id_headers_are_returned_for_success_and_errors(client):
    created = client.post("/v1/responses", headers={"x-request-id": "req_user_supplied"}, json={"input": "request id"})
    missing = client.get("/v1/responses/resp_missing", headers={"x-request-id": "req_missing"})

    assert created.status_code == 200
    assert created.headers["x-request-id"] == "req_user_supplied"
    assert missing.status_code == 404
    assert missing.headers["x-request-id"] == "req_missing"
    assert missing.json()["error"]["code"] == "not_found"


def test_idempotency_key_replays_same_response_and_rejects_conflicts(client):
    headers = {"Idempotency-Key": "phase14-idempotency"}
    payload = {"model": "gpt-oss-120b", "input": "idempotent create"}

    first = client.post("/v1/responses", headers=headers, json=payload)
    replayed = client.post("/v1/responses", headers=headers, json=payload)
    conflict = client.post("/v1/responses", headers=headers, json={**payload, "input": "changed"})
    empty_key = client.post("/v1/responses", headers={"Idempotency-Key": ""}, json=payload)

    assert first.status_code == 200
    assert replayed.status_code == 200
    assert replayed.headers["x-respawn-idempotent-replay"] == "true"
    assert replayed.json()["id"] == first.json()["id"]
    assert conflict.status_code == 409
    assert conflict.headers["x-request-id"].startswith("req_")
    assert conflict.json()["error"] == {
        "message": "Idempotency-Key was reused with a different request body.",
        "type": "invalid_request_error",
        "param": "Idempotency-Key",
        "code": "idempotency_conflict",
    }
    assert empty_key.status_code == 400
    assert empty_key.json()["error"]["param"] == "Idempotency-Key"


def test_error_schema_for_validation_and_unsupported_parameters(client):
    validation = client.post("/v1/responses", json={"input": "invalid", "temperature": 3})
    unsupported = client.post("/v1/responses", json={"input": "unsupported", "user": "legacy-user"})

    assert validation.status_code == 422
    assert validation.json()["error"]["type"] == "invalid_request_error"
    assert validation.json()["error"]["param"] == "temperature"
    assert validation.json()["error"]["code"] == "validation_error"
    assert unsupported.status_code == 400
    assert unsupported.json()["error"]["code"] == "unsupported_parameter"
    assert unsupported.json()["error"]["param"] == "user"


def test_response_request_settings_round_trip_through_retrieve(client):
    payload = {
        "model": "gpt-oss-120b",
        "input": "shape settings",
        "metadata": {"ticket": "settings-roundtrip"},
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
        assert body["metadata"] == {"ticket": "settings-roundtrip"}
        assert body["temperature"] == 0.2
        assert body["top_p"] == 0.9
        assert body["max_output_tokens"] == 16
        assert body["parallel_tool_calls"] is False
        assert body["service_tier"] == "default"
        assert body["text"] == {"format": {"type": "text"}}
        assert body["safety_identifier"] == "safe-local-user"
        assert body["store"] is True


def test_client_metadata_is_accepted_without_response_exposure(client):
    payload = {
        "model": "gpt-oss-120b",
        "input": "client metadata compatibility",
        "client_metadata": {"x-codex-installation-id": "install-local-test"},
        "store": True,
    }

    created = client.post("/v1/responses", json=payload)

    assert created.status_code == 200
    assert "client_metadata" not in created.json()
    retrieved = client.get(f"/v1/responses/{created.json()['id']}")
    assert retrieved.status_code == 200
    assert "client_metadata" not in retrieved.json()


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

    deleted = client.delete("/v1/responses/prompt_cache", params={"prompt_cache_key": "integration-cache"}).json()
    reset_counted = client.post("/v1/responses/input_tokens", json={**payload, "input": "shared prefix token token token token token variable-d"}).json()

    assert deleted["object"] == "prompt_cache.deleted"
    assert deleted["deleted"] > 0
    assert reset_counted["input_tokens_details"]["cached_tokens"] == 0


def test_prompt_template_render_variables_and_versions(client):
    first_template = client.post(
        "/v1/responses/prompts",
        json={
            "id": "pmpt_integration",
            "version": "1",
            "input": "Prompt template marker word {{word}}.",
            "metadata": {"case": "prompt-template"},
        },
    )
    assert first_template.status_code == 200
    assert first_template.json()["id"] == "pmpt_integration"
    assert first_template.json()["version"] == "1"

    second_template = client.post(
        "/v1/responses/prompts",
        json={
            "id": "pmpt_integration",
            "version": "2",
            "input": "Prompt template marker word sapphire.",
        },
    )
    assert second_template.status_code == 200

    rendered_v1 = client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "prompt": {"id": "pmpt_integration", "version": "1", "variables": {"word": "topaz"}},
        },
    ).json()
    rendered_latest = client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "prompt": {"id": "pmpt_integration"},
        },
    ).json()

    assert "topaz" in rendered_v1["output_text"]
    assert rendered_v1["input"] == "Prompt template marker word topaz."
    assert rendered_v1["prompt"]["version"] == "1"
    assert "sapphire" in rendered_latest["output_text"]
    assert rendered_latest["prompt"]["version"] == "2"


def test_prompt_template_missing_variable_returns_openai_error(client):
    client.post(
        "/v1/responses/prompts",
        json={"id": "pmpt_missing_variable", "version": "1", "input": "Prompt template marker word {{word}}."},
    )

    response = client.post("/v1/responses", json={"model": "gpt-oss-120b", "prompt": {"id": "pmpt_missing_variable"}})

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "missing_prompt_variable"
    assert body["error"]["param"] == "prompt.variables.word"


def test_prompt_template_missing_template_returns_openai_error(client):
    response = client.post("/v1/responses", json={"model": "gpt-oss-120b", "prompt": {"id": "pmpt_absent"}})

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "not_found"
    assert body["error"]["param"] == "prompt.id"


def test_prompt_template_tenant_isolation(tmp_path, monkeypatch):
    with configured_client(
        tmp_path,
        monkeypatch,
        DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'prompt_tenant.db'}",
        AUTH_DISABLED="false",
        LOCAL_OPENAI_API_KEYS="key-a:tenant-a,key-b:tenant-b",
    ) as client:
        created = client.post(
            "/v1/responses/prompts",
            headers={"Authorization": "Bearer key-a"},
            json={"id": "pmpt_tenant", "version": "1", "input": "Prompt template marker word tenant-a."},
        )
        allowed = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-a"},
            json={"model": "gpt-oss-120b", "prompt": {"id": "pmpt_tenant", "version": "1"}},
        )
        blocked = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-b"},
            json={"model": "gpt-oss-120b", "prompt": {"id": "pmpt_tenant", "version": "1"}},
        )

    assert created.status_code == 200
    assert allowed.status_code == 200
    assert "tenant-a" in allowed.json()["output_text"]
    assert blocked.status_code == 404
    assert blocked.json()["error"]["code"] == "not_found"
    assert blocked.json()["error"]["param"] == "prompt.id"


def test_truncation_disabled_overflow_fails_before_backend(tmp_path, monkeypatch):
    with configured_client(
        tmp_path,
        monkeypatch,
        DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'context_disabled.db'}",
        MODEL_CONTEXT_WINDOWS="gpt-oss-120b=80",
        CONTEXT_TOKEN_MARGIN="0",
        MAX_OUTPUT_TOKENS_DEFAULT="1",
    ) as client:
        response = client.post("/v1/responses", json={"input": " ".join(f"overflow-{index}" for index in range(120))})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"
    assert response.json()["error"]["param"] == "input"


def test_truncation_auto_drops_old_chain_and_records_event(tmp_path, monkeypatch):
    database_path = tmp_path / "context_auto.db"
    with configured_client(
        tmp_path,
        monkeypatch,
        DATABASE_URL=f"sqlite+aiosqlite:///{database_path}",
        MODEL_CONTEXT_WINDOWS="gpt-oss-120b=180",
        CONTEXT_TOKEN_MARGIN="0",
        MAX_OUTPUT_TOKENS_DEFAULT="1",
    ) as client:
        old_items = [{"role": "user", "content": f"old context item {index} " * 4} for index in range(40)]
        second = client.post(
            "/v1/responses",
            json={"input": [*old_items, {"role": "user", "content": "current short"}], "truncation": "auto", "store": True},
        )

    assert second.status_code == 200
    assert second.json()["truncation"] == "auto"
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("select type, strategy, input_tokens_before, input_tokens_after from response_context_events").fetchall()
    assert rows
    assert rows[0][0] == "truncation"
    assert rows[0][1] == "truncation_auto"
    assert rows[0][2] > rows[0][3]


def test_context_management_compaction_emits_item_and_preserves_fact(client):
    filler = " ".join(f"filler-{index}" for index in range(1200))
    first = client.post(
        "/v1/responses",
        json={"input": f"The preserved marker word is amethyst. {filler}", "store": True},
    ).json()

    second = client.post(
        "/v1/responses",
        json={
            "previous_response_id": first["id"],
            "input": "What is the preserved marker word?",
            "context_management": [{"type": "compaction", "compact_threshold": 1000}],
            "store": True,
        },
    )

    assert second.status_code == 200
    body = second.json()
    assert body["output"][0]["type"] == "compaction"
    assert isinstance(body["output"][0]["encrypted_content"], str)
    assert "amethyst" in body["output_text"]


def test_compact_endpoint_returns_compacted_window_and_followup_memory(client):
    filler = " ".join(f"compact-{index}" for index in range(80))
    compacted = client.post(
        "/v1/responses/compact",
        json={
            "input": [
                {"role": "user", "content": f"The preserved marker word is amethyst. {filler}"},
                {"role": "assistant", "content": "Noted."},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "The marker fact should be preserved."}]},
            ]
        },
    )

    assert compacted.status_code == 200
    compacted_body = compacted.json()
    assert compacted_body["object"] == "response.compaction"
    assert compacted_body["usage"]["input_tokens"] > 0
    compaction_item = compacted_body["output"][-1]
    assert compaction_item["type"] == "compaction"
    decoded = unseal_compaction_content(compaction_item["encrypted_content"], key=client.app.state.settings.reasoning_encryption_key)
    assert "amethyst" in decoded["summary"]["text"]
    assert "marker fact" in decoded["summary"]["text"]

    followup = client.post(
        "/v1/responses",
        json={"input": [*compacted_body["output"], {"role": "user", "content": "What is the preserved marker word?"}], "store": False},
    ).json()
    assert "amethyst" in followup["output_text"]


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


def test_reasoning_encrypted_content_round_trips_through_store_false(client):
    first = client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": "reason with encrypted content",
            "reasoning": {"effort": "low", "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    )

    assert first.status_code == 200
    first_body = first.json()
    reasoning_item = first_body["output"][0]
    assert reasoning_item["type"] == "reasoning"
    assert isinstance(reasoning_item["encrypted_content"], str)
    assert "mock backend inspected" not in json.dumps(reasoning_item).lower()
    decoded = unseal_reasoning_content(reasoning_item["encrypted_content"], key=client.app.state.settings.reasoning_encryption_key)
    assert "mock backend inspected" in decoded["reasoning"].lower()

    second = client.post(
        "/v1/responses",
        json={
            "input": [
                reasoning_item,
                {"role": "user", "content": "continue after encrypted reasoning"},
            ],
            "reasoning": {"effort": "low", "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
        },
    )

    assert second.status_code == 200
    second_body = second.json()
    assert second_body["output"][0]["type"] == "reasoning"
    assert isinstance(second_body["output"][0]["encrypted_content"], str)
    assert second_body["output_text"] == "Mock response: continue after encrypted reasoning"


def test_reasoning_encrypted_content_is_stored_and_retrieved(client):
    created = client.post(
        "/v1/responses",
        json={
            "input": "store encrypted reasoning",
            "reasoning": {"effort": "low", "summary": "auto"},
            "include": ["reasoning.encrypted_content"],
            "store": True,
        },
    ).json()

    retrieved = client.get(f"/v1/responses/{created['id']}").json()

    assert retrieved["output"][0]["type"] == "reasoning"
    assert retrieved["output"][0]["encrypted_content"] == created["output"][0]["encrypted_content"]


def test_reasoning_xhigh_requires_model_capability(client, tmp_path, monkeypatch):
    response = client.post("/v1/responses", json={"input": "xhigh please", "reasoning": {"effort": "xhigh"}})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_model_capability"
    assert response.json()["error"]["param"] == "reasoning.effort"

    with configured_client(
        tmp_path,
        monkeypatch,
        DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'xhigh.db'}",
        MODEL_CAPABILITIES="gpt-oss-120b=text,reasoning,reasoning-effort-xhigh",
    ) as xhigh_client:
        accepted = xhigh_client.post("/v1/responses", json={"input": "xhigh please", "reasoning": {"effort": "xhigh"}})

    assert accepted.status_code == 200
    assert accepted.json()["reasoning"]["effort"] == "xhigh"


def test_reasoning_effort_and_summary_values_are_validated(client):
    for effort in ("none", "minimal", "low", "medium", "high"):
        response = client.post("/v1/responses", json={"input": f"effort {effort}", "reasoning": {"effort": effort}})
        assert response.status_code == 200
        assert response.json()["reasoning"]["effort"] == effort

    for summary in ("auto", "concise", "detailed"):
        response = client.post("/v1/responses", json={"input": f"summary {summary}", "reasoning": {"effort": "low", "summary": summary}})
        assert response.status_code == 200
        assert response.json()["reasoning"]["summary"] == summary

    invalid_effort = client.post("/v1/responses", json={"input": "bad effort", "reasoning": {"effort": "extreme"}})
    assert invalid_effort.status_code == 400
    assert invalid_effort.json()["error"]["code"] == "unsupported_parameter"
    assert invalid_effort.json()["error"]["param"] == "reasoning.effort"

    invalid_summary = client.post("/v1/responses", json={"input": "bad summary", "reasoning": {"summary": "raw"}})
    assert invalid_summary.status_code == 400
    assert invalid_summary.json()["error"]["code"] == "unsupported_parameter"
    assert invalid_summary.json()["error"]["param"] == "reasoning.summary"


def test_include_registry_rejects_hosted_tool_and_unknown_expansions(client):
    response = client.post("/v1/responses", json={"input": "hello", "include": ["file_search_call.results"]})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == "include.0"

    unknown = client.post("/v1/responses", json={"input": "hello", "include": ["message.input_file.artifact"]})
    assert unknown.status_code == 400
    assert unknown.json()["error"]["code"] == "unsupported_parameter"
    assert unknown.json()["error"]["param"] == "include.0"


def test_output_logprobs_include_requires_model_capability(client):
    response = client.post("/v1/responses", json={"input": "hello", "include": ["message.output_text.logprobs"]})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_model_capability"
    assert response.json()["error"]["param"] == "include"


def test_output_logprobs_are_returned_when_mock_backend_can_provide_them(tmp_path, monkeypatch):
    with configured_client(
        tmp_path,
        monkeypatch,
        MODEL_CAPABILITIES="gpt-oss-120b=text,file-text,reasoning,tools,logprobs;moondream:latest=text,file-text,vision",
    ) as client:
        created = client.post(
            "/v1/responses",
            json={"input": "logprob marker", "include": ["message.output_text.logprobs"], "top_logprobs": 2, "store": True},
        )

        assert created.status_code == 200
        body = created.json()
        logprobs = body["output"][0]["content"][0]["logprobs"]
        assert logprobs
        assert {"token", "bytes", "logprob", "top_logprobs"}.issubset(logprobs[0])
        assert len(logprobs[0]["top_logprobs"]) <= 3

        retrieved = client.get(f"/v1/responses/{body['id']}").json()
        assert retrieved["output"][0]["content"][0]["logprobs"] == logprobs


def test_retrieve_include_can_expand_stored_logprobs(tmp_path, monkeypatch):
    with configured_client(
        tmp_path,
        monkeypatch,
        MODEL_CAPABILITIES="gpt-oss-120b=text,file-text,reasoning,tools,logprobs;moondream:latest=text,file-text,vision",
    ) as client:
        created = client.post("/v1/responses", json={"input": "deferred logprobs", "top_logprobs": 1, "store": True}).json()
        assert created["output"][0]["content"][0]["logprobs"] == []

        retrieved = client.get(f"/v1/responses/{created['id']}?include[]=message.output_text.logprobs").json()
        assert retrieved["output"][0]["content"][0]["logprobs"]


def test_input_file_artifacts_produce_output_annotations_and_retrieve(client):
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

    annotation = created["output"][0]["content"][0]["annotations"][0]
    assert annotation["type"] == "file_citation"
    assert annotation["file_id"].startswith("art_")
    assert annotation["filename"] == "facts.txt"
    retrieved = client.get(f"/v1/responses/{created['id']}").json()
    assert retrieved["output"][0]["content"][0]["annotations"] == created["output"][0]["content"][0]["annotations"]
    input_items = client.get(f"/v1/responses/{created['id']}/input_items?order=asc").json()["data"]
    assert not any(key.startswith("_respawn_") for key in input_items[0]["content"][0])
    artifact_content = client.get(f"/v1/responses/{created['id']}/artifacts/{annotation['file_id']}/content")
    assert artifact_content.status_code == 200
    assert artifact_content.text == file_text


def test_response_artifacts_list_paginates(client):
    first_text = "Respawn first artifact marker."
    second_text = "Respawn second artifact marker."
    created = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "filename": "first.txt", "file_data": _data_url("text/plain", base64.b64encode(first_text.encode()).decode())},
                        {"type": "input_file", "filename": "second.txt", "file_data": _data_url("text/plain", base64.b64encode(second_text.encode()).decode())},
                        {"type": "input_text", "text": "Mention both markers."},
                    ],
                }
            ],
            "store": True,
        },
    ).json()

    first_page = client.get(f"/v1/responses/{created['id']}/artifacts?order=asc&limit=1").json()
    second_page = client.get(f"/v1/responses/{created['id']}/artifacts?order=asc&limit=1&after={first_page['data'][0]['id']}").json()
    before_second = client.get(f"/v1/responses/{created['id']}/artifacts?order=asc&before={second_page['data'][0]['id']}&limit=10").json()

    assert first_page["object"] == "list"
    assert first_page["has_more"] is True
    assert first_page["first_id"] == first_page["data"][0]["id"]
    assert first_page["last_id"] == first_page["data"][0]["id"]
    assert first_page["data"][0]["filename"] == "first.txt"
    assert second_page["data"][0]["filename"] == "second.txt"
    assert before_second["data"][0]["id"] == first_page["data"][0]["id"]


def test_files_api_create_list_content_delete(client):
    upload = _upload_text_file(client, filename="facts-a.txt")
    upload_b = _upload_text_file(client, filename="facts-b.txt", text="Respawn second file marker.")

    assert upload.status_code == 200
    assert upload_b.status_code == 200
    created = upload.json()
    created_b = upload_b.json()
    assert created["object"] == "file"
    assert created["id"].startswith("file_")
    assert created["bytes"] == len("Respawn file marker word: cobalt.".encode())
    assert created["purpose"] == "user_data"

    listed = client.get("/v1/files?order=asc&limit=1").json()
    assert [item["id"] for item in listed["data"]] == [created["id"]]
    assert listed["first_id"] == created["id"]
    assert listed["last_id"] == created["id"]
    assert listed["has_more"] is True
    second_page = client.get(f"/v1/files?order=asc&after={created['id']}&limit=1").json()
    assert [item["id"] for item in second_page["data"]] == [created_b["id"]]

    retrieved = client.get(f"/v1/files/{created['id']}").json()
    content = client.get(f"/v1/files/{created['id']}/content")

    assert retrieved["filename"] == "facts-a.txt"
    assert content.status_code == 200
    assert content.text == "Respawn file marker word: cobalt."

    deleted = client.delete(f"/v1/files/{created['id']}").json()
    missing = client.get(f"/v1/files/{created['id']}")

    assert deleted == {"id": created["id"], "object": "file", "deleted": True}
    assert missing.status_code == 404


def test_files_api_quota_and_malware_errors(tmp_path, monkeypatch):
    with configured_client(
        tmp_path,
        monkeypatch,
        DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'files_errors.db'}",
        FILE_STORAGE_QUOTA_BYTES="8",
    ) as client:
        quota = client.post(
            "/v1/files",
            data={"purpose": "user_data"},
            files={"file": ("too-large.txt", b"123456789", "text/plain")},
        )

    with configured_client(
        tmp_path,
        monkeypatch,
        DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'files_malware.db'}",
    ) as client:
        malware = client.post(
            "/v1/files",
            data={"purpose": "user_data"},
            files={"file": ("eicar.txt", b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR", "text/plain")},
        )

    assert quota.status_code == 400
    assert quota.json()["error"]["code"] == "storage_quota_exceeded"
    assert malware.status_code == 400
    assert malware.json()["error"]["code"] == "file_malware_detected"


def test_files_api_ttl_cleanup(tmp_path, monkeypatch):
    with configured_client(
        tmp_path,
        monkeypatch,
        DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'files_ttl.db'}",
        FILE_DEFAULT_TTL_SECONDS="1",
        FILE_CLEANUP_INTERVAL_SECONDS="0.1",
    ) as client:
        uploaded = _upload_text_file(client).json()
        assert client.get(f"/v1/files/{uploaded['id']}").status_code == 200
        time.sleep(1.4)
        expired = client.get(f"/v1/files/{uploaded['id']}")

    assert expired.status_code == 404


def test_input_image_include_returns_safe_artifact_metadata(client):
    created = client.post(
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
            "include": ["message.input_image.image_url"],
            "store": True,
        },
    )

    assert created.status_code == 200
    body = created.json()
    image_part = body["input"][0]["content"][1]
    assert image_part["artifact"]["id"].startswith("art_")
    assert image_part["artifact"]["source"] == {"type": "data_url", "redacted": True}
    assert "content" not in image_part["artifact"]

    retrieved = client.get(f"/v1/responses/{body['id']}?include=message.input_image.image_url").json()
    assert retrieved["input"][0]["content"][1]["artifact"] == image_part["artifact"]


def test_background_response_create_poll_complete(client):
    created = client.post(
        "/v1/responses",
        json={"input": "background complete", "background": True, "store": True, "metadata": mock_metadata(delay_seconds=0.15)},
    )

    assert created.status_code == 200
    body = created.json()
    assert body["background"] is True
    assert body["status"] in {"queued", "in_progress", "completed"}
    if body["status"] != "completed":
        assert body["output"] == []

    completed = poll_response(client, body["id"], expected={"completed"})

    assert completed["background"] is True
    assert completed["output_text"] == "Mock response: background complete"
    assert completed["usage"]["total_tokens"] > 0


def test_background_cancel_is_terminal_and_idempotent(client):
    created = client.post(
        "/v1/responses",
        json={"input": "background cancel", "background": True, "store": True, "metadata": mock_metadata(delay_seconds=0.15)},
    ).json()

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
        created = client.post(
            "/v1/responses",
            json={"input": "background delay fixture", "background": True, "store": True, "metadata": mock_metadata(delay_seconds=0.25)},
        ).json()
        failed = poll_response(client, created["id"], expected={"failed"})

    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "background_timeout"


def test_background_metrics_include_job_signals(client):
    created = client.post(
        "/v1/responses",
        json={"input": "background metrics", "background": True, "store": True, "metadata": mock_metadata(delay_seconds=0.05)},
    ).json()
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

    followup = client.post("/v1/responses", json={"previous_response_id": created["id"], "input": "Use the previous file fact."}).json()
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


def test_file_id_input_file_is_resolved_and_deleted_files_fail(client):
    uploaded = _upload_text_file(client).json()
    created = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": uploaded["id"]},
                        {"type": "input_text", "text": "Repeat the marker word."},
                    ],
                }
            ],
            "store": True,
        },
    )

    assert created.status_code == 200
    body = created.json()
    assert "cobalt" in body["output_text"]
    input_items = client.get(f"/v1/responses/{body['id']}/input_items?order=asc").json()["data"]
    file_part = input_items[0]["content"][0]
    assert file_part["file_id"] == uploaded["id"]
    assert file_part["text"] == "Respawn file marker word: cobalt."

    client.delete(f"/v1/files/{uploaded['id']}")
    missing = client.post("/v1/responses", json={"input": [{"role": "user", "content": [{"type": "input_file", "file_id": uploaded["id"]}]}]})
    assert missing.status_code == 404
    assert missing.json()["error"]["param"] == "input.0.content.0.file_id"


def test_file_id_tenant_isolation(tmp_path, monkeypatch):
    with configured_client(
        tmp_path,
        monkeypatch,
        DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'files_tenant.db'}",
        AUTH_DISABLED="false",
        LOCAL_OPENAI_API_KEYS="key-a:tenant-a,key-b:tenant-b",
    ) as client:
        uploaded = _upload_text_file(client, headers={"Authorization": "Bearer key-a"}).json()
        allowed = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-a"},
            json={"input": [{"role": "user", "content": [{"type": "input_file", "file_id": uploaded["id"]}, {"type": "input_text", "text": "Repeat marker."}]}]},
        )
        blocked_retrieve = client.get(f"/v1/files/{uploaded['id']}", headers={"Authorization": "Bearer key-b"})
        blocked_response = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer key-b"},
            json={"input": [{"role": "user", "content": [{"type": "input_file", "file_id": uploaded["id"]}]}]},
        )

    assert allowed.status_code == 200
    assert "cobalt" in allowed.json()["output_text"]
    assert blocked_retrieve.status_code == 404
    assert blocked_response.status_code == 404
    assert blocked_response.json()["error"]["param"] == "input.0.content.0.file_id"


def test_audio_inputs_are_explicitly_unsupported(client):

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

    agent_like_response = client.post(
        "/v1/responses",
        json={
            "input": "search",
            "client_metadata": {"x-codex-installation-id": "install-local-test"},
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Run a command.",
                    "parameters": {"type": "object", "properties": {}},
                },
                {"type": "web_search"},
            ],
        },
    )

    assert agent_like_response.status_code == 400
    assert agent_like_response.json()["error"]["code"] == "unsupported_parameter"
    assert agent_like_response.json()["error"]["param"] == "tools.1.type"


def test_web_search_required_returns_call_citations_and_retrieves(web_search_client):
    response = web_search_client.post(
        "/v1/responses",
        json={
            "input": "Search the web for latest Respawn web search news",
            "tools": [{"type": "web_search"}],
            "tool_choice": "required",
            "store": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["output"][0]["type"] == "web_search_call"
    assert body["output"][0]["action"]["type"] == "search"
    assert body["output"][0]["action"]["queries"] == ["Search the web for latest Respawn web search news"]
    assert "sources" not in body["output"][0]["action"]
    assert body["output"][1]["type"] == "message"
    annotations = body["output"][1]["content"][0]["annotations"]
    assert annotations
    assert annotations[0]["type"] == "url_citation"
    assert body["output_text"].startswith("Mock response:")

    retrieved = web_search_client.get(f"/v1/responses/{body['id']}").json()
    assert retrieved["output"][0]["type"] == "web_search_call"
    assert retrieved["output"][1]["content"][0]["annotations"] == annotations

    included = web_search_client.get(
        f"/v1/responses/{body['id']}",
        params={"include": "web_search_call.action.sources"},
    ).json()
    sources = included["output"][0]["action"]["sources"]
    assert sources
    assert sources[0]["url"] == "https://example.com/respawn-web-search"


def test_web_search_auto_uses_model_tool_call(web_search_client):
    response = web_search_client.post(
        "/v1/responses",
        json={
            "input": "Search the web for latest Respawn auto search details",
            "tools": [{"type": "web_search"}],
            "metadata": mock_metadata(tool_call="web_search"),
            "store": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output"][0]["type"] == "web_search_call"
    assert body["output"][1]["type"] == "message"
    assert all(item["type"] != "function_call" for item in body["output"])
    assert body["output"][1]["content"][0]["annotations"][0]["type"] == "url_citation"
    assert web_search_client.app.state.web_search_backend.requests[0].query == "Search the web for latest Respawn auto search details"


def test_web_search_followup_is_text_only_when_image_generation_is_available(web_search_and_image_generation_client):
    response = web_search_and_image_generation_client.post(
        "/v1/responses",
        json={
            "input": "Search externally for the current Respawn routing fixture",
            "tools": [{"type": "web_search"}, {"type": "image_generation", "quality": "low"}],
            "metadata": mock_metadata(tool_call="web_search"),
            "store": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["type"] for item in body["output"]] == ["web_search_call", "message"]
    assert web_search_and_image_generation_client.app.state.image_generation_backend.requests == []

    payloads = web_search_and_image_generation_client.app.state.backend.payloads
    assert "respawn_web_search" in backend_tool_names(payloads[0])
    assert "respawn_image_generation" in backend_tool_names(payloads[0])
    assert payloads[0]["messages"][0]["content"].startswith("Respawn tool-use policy:")
    assert "answer normally without calling web_search" in payloads[0]["messages"][0]["content"]
    assert "tools" not in payloads[1]
    assert "tool_choice" not in payloads[1]
    assert all(not message.get("content", "").startswith("Respawn tool-use policy:") for message in payloads[1]["messages"] if message.get("role") == "system")


def test_image_generation_auto_keeps_tool_available_after_prior_image(image_generation_client):
    first = image_generation_client.post(
        "/v1/responses",
        json={
            "input": "Generate image of an auto tiny house",
            "tools": [{"type": "image_generation", "quality": "low"}],
            "metadata": mock_metadata(tool_call="image_generation"),
            "store": True,
        },
    )
    assert first.status_code == 200
    assert first.json()["output"][0]["type"] == "image_generation_call"

    response = image_generation_client.post(
        "/v1/responses",
        json={
            "previous_response_id": first.json()["id"],
            "input": "Continue with a normal text answer.",
            "tools": [{"type": "image_generation", "quality": "low"}],
            "store": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert all(item["type"] != "image_generation_call" for item in body["output"])
    assert len(image_generation_client.app.state.image_generation_backend.requests) == 1
    payload = image_generation_client.app.state.backend.payloads[-1]
    assert "respawn_image_generation" in backend_tool_names(payload)
    assert "do not show an image placeholder" in payload["messages"][0]["content"]


def test_image_generation_auto_can_generate_consecutive_images(image_generation_client):
    first = image_generation_client.post(
        "/v1/responses",
        json={
            "input": "Generate image of an auto tiny house",
            "tools": [{"type": "image_generation", "quality": "low"}],
            "metadata": mock_metadata(tool_call="image_generation"),
            "store": True,
        },
    )
    assert first.status_code == 200
    assert first.json()["output"][0]["type"] == "image_generation_call"

    second = image_generation_client.post(
        "/v1/responses",
        json={
            "previous_response_id": first.json()["id"],
            "input": "Generate image of a tiny mouse",
            "tools": [{"type": "image_generation", "quality": "low"}],
            "metadata": mock_metadata(tool_call="image_generation"),
            "store": False,
        },
    )

    assert second.status_code == 200
    body = second.json()
    assert body["output"][0]["type"] == "image_generation_call"
    assert len(image_generation_client.app.state.image_generation_backend.requests) == 2
    payload = image_generation_client.app.state.backend.payloads[-1]
    assert "respawn_image_generation" in backend_tool_names(payload)


def test_image_generation_auto_reexposes_after_intervening_text_context(image_generation_client):
    response = image_generation_client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "type": "image_generation_call",
                    "status": "completed",
                    "revised_prompt": "a toad",
                    "result": "base64-png",
                    "size": "512x512",
                },
                {"type": "message", "role": "user", "content": "What is Kubernetes?"},
                {"type": "message", "role": "assistant", "content": "Kubernetes orchestrates containers."},
                {"type": "message", "role": "user", "content": "Generate image of a dog"},
            ],
            "tools": [{"type": "image_generation", "quality": "low"}],
            "metadata": mock_metadata(tool_call="image_generation"),
            "store": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output"][0]["type"] == "image_generation_call"
    assert len(image_generation_client.app.state.image_generation_backend.requests) == 1
    payload = image_generation_client.app.state.backend.payloads[-1]
    assert "respawn_image_generation" in backend_tool_names(payload)


def test_image_generation_auto_keeps_tool_after_empty_assistant_after_prior_image_context(image_generation_client):
    response = image_generation_client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "type": "image_generation_call",
                    "status": "completed",
                    "revised_prompt": "a dog",
                    "result": "base64-png",
                    "size": "512x512",
                },
                {"type": "message", "role": "assistant", "content": ""},
                {"type": "message", "role": "user", "content": "What is Kubernetes?"},
            ],
            "tools": [{"type": "image_generation", "quality": "low"}],
            "store": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert all(item["type"] != "image_generation_call" for item in body["output"])
    assert image_generation_client.app.state.image_generation_backend.requests == []
    payload = image_generation_client.app.state.backend.payloads[-1]
    assert "respawn_image_generation" in backend_tool_names(payload)


def test_web_search_tool_choice_none_does_not_call_provider(web_search_client):
    response = web_search_client.post(
        "/v1/responses",
        json={
            "input": "What is the latest Respawn news?",
            "tools": [{"type": "web_search"}],
            "tool_choice": "none",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output"][0]["type"] == "message"
    assert all(item["type"] != "web_search_call" for item in body["output"])
    assert web_search_client.app.state.web_search_backend.requests == []


def test_web_search_filters_store_false_and_timeout(web_search_client):
    filtered = web_search_client.post(
        "/v1/responses",
        json={
            "input": "Search the web for latest domain filter details",
            "tools": [{"type": "web_search", "filters": {"blocked_domains": ["blocked.example.net"]}}],
            "tool_choice": "required",
            "include": ["web_search_call.action.sources"],
            "store": False,
        },
    )
    assert filtered.status_code == 200
    body = filtered.json()
    assert body["output"][0]["type"] == "web_search_call"
    assert all("blocked.example.net" not in source["url"] for source in body["output"][0]["action"]["sources"])
    assert web_search_client.get(f"/v1/responses/{body['id']}").status_code == 404

    timeout = web_search_client.post(
        "/v1/responses",
        json={
            "input": "Search provider timeout fixture",
            "tools": [{"type": "web_search"}],
            "tool_choice": "required",
            "metadata": mock_metadata(web_search_error="timeout"),
        },
    )
    assert timeout.status_code == 504
    assert timeout.json()["error"]["code"] == "web_search_timeout"


def test_agent_like_namespace_plus_web_search_passes_web_search_validation(web_search_client):
    accepted = web_search_client.post(
        "/v1/responses",
        json={
            "input": "search",
            "client_metadata": {"x-codex-installation-id": "install-local-test"},
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__repo__",
                    "tools": [
                        {
                            "type": "function",
                            "name": "list_files",
                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                        }
                    ],
                },
                {"type": "web_search"},
            ],
            "tool_choice": "none",
        },
    )
    assert accepted.status_code == 200

    next_unsupported_tool = web_search_client.post(
        "/v1/responses",
        json={
            "input": "search",
            "tools": [
                {"type": "web_search"},
                {"type": "image_generation"},
            ],
        },
    )
    assert next_unsupported_tool.status_code == 400
    assert next_unsupported_tool.json()["error"]["param"] == "tools.1.type"


def test_image_generation_disabled_returns_openai_shaped_error(client):
    response = client.post("/v1/responses", json={"input": "Generate image of a tiny house", "tools": [{"type": "image_generation"}]})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == "tools.0.type"


def test_image_generation_auto_uses_model_tool_call(image_generation_client):
    response = image_generation_client.post(
        "/v1/responses",
        json={
            "input": "Generate image of an auto tiny house",
            "tools": [{"type": "image_generation", "quality": "low"}],
            "metadata": mock_metadata(tool_call="image_generation"),
            "store": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output_text"] == ""
    assert body["output"][0]["type"] == "image_generation_call"
    assert all(item["type"] != "function_call" for item in body["output"])
    assert image_generation_client.app.state.image_generation_backend.requests[0].prompt == "Generate image of an auto tiny house"


def test_image_generation_forced_returns_call_and_retrieves(image_generation_client):
    response = image_generation_client.post(
        "/v1/responses",
        json={
            "input": "Generate image of a tiny house",
            "tools": [{"type": "image_generation", "quality": "low", "size": "512x512"}],
            "tool_choice": {"type": "image_generation"},
            "store": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["output_text"] == ""
    assert body["output"][0]["type"] == "image_generation_call"
    assert body["output"][0]["status"] == "completed"
    assert body["output"][0]["result"]
    assert body["output"][0]["size"] == "512x512"
    assert body["output"][0]["quality"] == "low"
    assert body["output"][0]["output_format"] == "png"

    retrieved = image_generation_client.get(f"/v1/responses/{body['id']}").json()
    assert retrieved["output"] == body["output"]

    required = image_generation_client.post(
        "/v1/responses",
        json={
            "input": "Generate image of a required tiny house",
            "tools": [{"type": "image_generation"}],
            "tool_choice": "required",
            "store": False,
        },
    )
    assert required.status_code == 200
    required_body = required.json()
    assert required_body["output"][0]["type"] == "image_generation_call"
    assert image_generation_client.get(f"/v1/responses/{required_body['id']}").status_code == 404


def test_image_generation_followup_accepts_prior_web_search_call_item(image_generation_client):
    response = image_generation_client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "id": "ws_prior",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "queries": ["cos'e kubernetes"],
                        "sources": [
                            {
                                "url": "https://kubernetes.io/",
                                "title": "Kubernetes",
                                "snippet": "Production-grade container orchestration.",
                            }
                        ],
                    },
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Kubernetes orchestra container."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "genera una immagine di un rospo"}],
                },
            ],
            "tools": [{"type": "image_generation", "quality": "low"}],
            "tool_choice": {"type": "image_generation"},
            "store": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output"][0]["type"] == "image_generation_call"
    assert body["output"][0]["status"] == "completed"


def test_image_generation_tool_choice_none_and_agent_like_declaration_do_not_generate(image_generation_client):
    agent_like = image_generation_client.post(
        "/v1/responses",
        json={
            "input": "List files in this repository",
            "client_metadata": {"x-codex-installation-id": "install-local-test"},
            "tools": [{"type": "image_generation"}],
        },
    )
    assert agent_like.status_code == 200
    assert all(item["type"] != "image_generation_call" for item in agent_like.json()["output"])
    assert image_generation_client.app.state.image_generation_backend.requests == []

    disabled_by_choice = image_generation_client.post(
        "/v1/responses",
        json={
            "input": "Generate image of a tiny house",
            "tools": [{"type": "image_generation"}],
            "tool_choice": "none",
        },
    )
    assert disabled_by_choice.status_code == 200
    assert all(item["type"] != "image_generation_call" for item in disabled_by_choice.json()["output"])


def test_image_generation_background_and_metrics(image_generation_client):
    created = image_generation_client.post(
        "/v1/responses",
        json={
            "input": "Generate image of a tiny background house",
            "tools": [{"type": "image_generation"}],
            "tool_choice": {"type": "image_generation"},
            "background": True,
            "store": True,
        },
    )
    assert created.status_code == 200
    body = created.json()
    completed = poll_response(image_generation_client, body["id"], expected={"completed"})
    assert completed["background"] is True
    assert completed["output"][0]["type"] == "image_generation_call"

    timeout = image_generation_client.post(
        "/v1/responses",
        json={
            "input": "Image provider timeout fixture",
            "tools": [{"type": "image_generation"}],
            "tool_choice": {"type": "image_generation"},
            "metadata": mock_metadata(image_generation_error="timeout"),
        },
    )
    assert timeout.status_code == 504
    assert timeout.json()["error"]["code"] == "image_generation_timeout"

    metrics = image_generation_client.get("/metrics").text
    assert "gateway_image_generation_requests_total" in metrics
    assert "gateway_image_generation_latency_seconds_bucket" in metrics
    assert "gateway_image_generation_errors_total" in metrics
    assert "gateway_image_generation_pixels_total" in metrics


def test_web_search_metrics_are_exposed(web_search_client):
    web_search_client.post(
        "/v1/responses",
        json={
            "input": "Search the web for latest metrics web search details",
            "tools": [{"type": "web_search"}],
            "tool_choice": "required",
        },
    )
    web_search_client.post(
        "/v1/responses",
        json={
            "input": "Search the web for latest filtered metrics details",
            "tools": [{"type": "web_search", "filters": {"blocked_domains": ["blocked.example.net"]}}],
            "tool_choice": "required",
        },
    )
    web_search_client.post(
        "/v1/responses",
        json={
            "input": "Metrics web search failure fixture",
            "tools": [{"type": "web_search"}],
            "tool_choice": "required",
            "metadata": mock_metadata(web_search_error="timeout"),
        },
    )

    metrics = web_search_client.get("/metrics").text
    assert "gateway_web_search_requests_total" in metrics
    assert "gateway_web_search_latency_seconds_bucket" in metrics
    assert "gateway_web_search_results_total" in metrics
    assert "gateway_web_search_errors_total" in metrics
    assert "gateway_web_search_filtered_results_total" in metrics


def test_invalid_prompt_cache_retention_is_explicit(client):
    response = client.post("/v1/responses", json={"input": "hello", "prompt_cache_retention": "forever"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == "prompt_cache_retention"


def test_metrics_include_model_gateway_signals(client):
    client.get("/readyz")
    client.post("/v1/responses", json={"model": "gpt-oss-120b", "input": "metrics"})
    client.post("/v1/responses", json={"model": "gpt-oss-120b", "input": "metrics idempotent"}, headers={"Idempotency-Key": "metrics-idempotent"})
    client.post("/v1/responses", json={"model": "gpt-oss-120b", "input": "metrics idempotent"}, headers={"Idempotency-Key": "metrics-idempotent"})
    client.post("/v1/responses/prompts", json={"id": "pmpt_metrics", "version": "1", "input": "Prompt template marker word metrics."})
    client.post("/v1/responses", json={"model": "gpt-oss-120b", "prompt": {"id": "pmpt_metrics", "version": "1"}})
    uploaded = _upload_text_file(client).json()
    client.get(f"/v1/files/{uploaded['id']}/content")
    client.delete(f"/v1/files/{uploaded['id']}")
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
    with client.stream("POST", "/v1/responses", json={"model": "gpt-oss-120b", "input": "metrics stream", "stream": True}) as stream:
        "".join(stream.iter_text())

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "gateway_responses_total" in response.text
    assert "gateway_response_latency_seconds_bucket" in response.text
    assert "gateway_endpoint_requests_total" in response.text
    assert "gateway_feature_requests_total" in response.text
    assert "gateway_idempotency_requests_total" in response.text
    assert "gateway_inflight_responses" in response.text
    assert "gateway_model_token_usage_total" in response.text
    assert "gateway_backend_model_info" in response.text
    assert "gateway_backend_model_requests_total" in response.text
    assert "gateway_backend_eval_tokens_total" in response.text
    assert "gateway_backend_eval_duration_seconds_total" in response.text
    assert "gateway_backend_eval_tokens_per_second" in response.text
    assert "gateway_operational_failures_total" in response.text
    assert "gateway_readiness_check" in response.text
    assert "gateway_readiness_check_latency_seconds_bucket" in response.text
    assert "gateway_storage_operations_total" in response.text
    assert "gateway_streaming_responses_running" in response.text
    assert "gateway_function_tool_requests_total" in response.text
    assert "gateway_function_tool_calls_total" in response.text
    assert "gateway_function_tool_outputs_total" in response.text
    assert "gateway_prompt_template_requests_total" in response.text
    assert "gateway_prompt_cache_requests_total" in response.text
    assert "gateway_prompt_cache_hit_ratio" in response.text


def test_request_logs_include_operational_fields(client, monkeypatch):
    log_records = []

    def capture_log(message, *_, **kwargs):
        log_records.append((message, kwargs.get("extra") or {}))

    monkeypatch.setattr("src.main.logger.info", capture_log)
    response = client.post("/v1/responses", json={"model": "gpt-oss-120b", "input": "structured log fields"})

    assert response.status_code == 200
    records = [record for record in log_records if record[0] == "HTTP request completed"]
    assert records
    extra = records[-1][1]
    assert extra["request_id"] == response.headers["x-request-id"]
    assert extra["response_id"] == response.json()["id"]
    assert extra["tenant"] is None
    assert extra["feature"] == "responses"
    assert extra["backend"] == "mock"
    assert extra["status"] == 200
    assert extra["error_code"] is None
    assert extra["error_param"] is None
    assert extra["latency_ms"] >= 0


def test_request_logs_include_error_param(client, monkeypatch):
    log_records = []

    def capture_info(message, **kwargs):
        log_records.append((message, kwargs.get("extra") or {}))

    monkeypatch.setattr("src.main.logger.info", capture_info)

    response = client.post("/v1/responses", json={"input": "bad", "client_metadata": 1})

    assert response.status_code == 422
    records = [record for record in log_records if record[0] == "HTTP request completed"]
    extra = records[-1][1]
    assert extra["error_code"] == "validation_error"
    assert extra["error_param"] == "client_metadata"


def test_metrics_request_log_is_trace_noise(client, monkeypatch):
    info_records = []
    debug_records = []
    log_records = []

    def capture_info(message, **kwargs):
        info_records.append((message, kwargs.get("extra") or {}))

    def capture_debug(message, **kwargs):
        debug_records.append((message, kwargs.get("extra") or {}))

    def capture_log(level, message, **kwargs):
        log_records.append((level, message, kwargs.get("extra") or {}))

    monkeypatch.setattr("src.main.logger.info", capture_info)
    monkeypatch.setattr("src.main.logger.debug", capture_debug)
    monkeypatch.setattr("src.main.logger.log", capture_log)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert not [record for record in info_records if record[0] == "HTTP request completed"]
    assert not [record for record in debug_records if record[0] == "HTTP request completed"]
    records = [record for record in log_records if record[0] == TRACE_LEVEL and record[1] == "HTTP request completed"]
    assert records
    assert records[-1][2]["feature"] == "metrics"


def test_reasoning_metrics_include_effort_tokens_and_heavy_counter(tmp_path, monkeypatch):
    with configured_client(
        tmp_path,
        monkeypatch,
        DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'reasoning_metrics.db'}",
        REASONING_HEAVY_TOKEN_THRESHOLD="1",
    ) as client:
        client.post(
            "/v1/responses",
            json={"model": "gpt-oss-120b", "input": "metrics reasoning", "reasoning": {"effort": "low", "summary": "auto"}},
        )
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "gateway_reasoning_requests_total" in response.text
    assert "gateway_reasoning_tokens_total" in response.text
    assert "gateway_reasoning_heavy_requests_total" in response.text


def test_context_metrics_include_compaction_and_truncation(client):
    client.post("/v1/responses/compact", json={"input": "The preserved marker word is amethyst."})

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "gateway_context_compactions_total" in response.text
    assert "gateway_context_compaction_tokens_total" in response.text
    assert "gateway_context_compaction_ratio_bucket" in response.text
    assert "gateway_context_truncations_total" in response.text
    assert "gateway_context_overflows_total" in response.text


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
            "arguments": '{"expression":"string"}',
        }
    ]


def test_custom_tool_request_emits_custom_tool_call_without_executing(client):
    response = client.post(
        "/v1/responses",
        json={
            "input": "Generate raw HTML for a small page",
            "tools": [
                {
                    "type": "custom",
                    "name": "html_writer",
                    "description": "Write raw HTML.",
                    "format": {"type": "text"},
                }
            ],
            "tool_choice": {"type": "custom", "name": "html_writer"},
            "store": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output_text"] == ""
    call = body["output"][0]
    assert call["type"] == "custom_tool_call"
    assert call["status"] == "completed"
    assert call["call_id"] == "call_mock_html_writer"
    assert call["name"] == "html_writer"
    assert call["input"] == "string"

    input_items = client.get(f"/v1/responses/{body['id']}/input_items?order=asc").json()["data"]
    assert input_items[0]["content"][0]["text"] == "Generate raw HTML for a small page"


def test_custom_tool_call_output_followup_with_previous_response_id(client):
    first = client.post(
        "/v1/responses",
        json={
            "input": "Generate raw HTML please",
            "tools": [{"type": "custom", "name": "html_writer"}],
            "tool_choice": {"type": "custom", "name": "html_writer"},
            "store": True,
        },
    ).json()
    call = first["output"][0]

    second = client.post(
        "/v1/responses",
        json={
            "previous_response_id": first["id"],
            "input": [{"type": "custom_tool_call_output", "call_id": call["call_id"], "output": "<h1>Cobalt</h1>"}],
            "tools": [{"type": "custom", "name": "html_writer"}],
            "store": True,
        },
    )

    assert second.status_code == 200
    body = second.json()
    assert body["output"][0]["type"] == "message"
    assert body["output_text"] == "Tool result: <h1>Cobalt</h1>"
    input_items = client.get(f"/v1/responses/{body['id']}/input_items?order=asc").json()["data"]
    assert input_items[0]["type"] == "custom_tool_call_output"
    assert input_items[0]["call_id"] == call["call_id"]


def test_namespace_custom_tool_request_emits_namespaced_custom_tool_call(client):
    response = client.post(
        "/v1/responses",
        json={
            "input": "Use namespace custom tool please",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__repo__",
                    "description": "Repository tools.",
                    "tools": [
                        {
                            "type": "custom",
                            "name": "write_patch",
                            "description": "Write a raw patch.",
                            "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
                        }
                    ],
                }
            ],
            "tool_choice": {"type": "custom", "namespace": "mcp__repo__", "name": "write_patch"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output_text"] == ""
    assert body["output"][0]["type"] == "custom_tool_call"
    assert body["output"][0]["name"] == "write_patch"
    assert body["output"][0]["namespace"] == "mcp__repo__"


def test_namespace_tool_request_emits_namespaced_function_call(client):
    response = client.post(
        "/v1/responses",
        json={
            "input": "Use namespace tool please",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__repo__",
                    "description": "Repository tools.",
                    "tools": [
                        {
                            "type": "function",
                            "name": "list_files",
                            "description": "List files in a path.",
                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                        }
                    ],
                }
            ],
            "tool_choice": {"type": "function", "namespace": "mcp__repo__", "name": "list_files"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["output_text"] == ""
    assert body["output"][0]["type"] == "function_call"
    assert body["output"][0]["name"] == "list_files"
    assert body["output"][0]["namespace"] == "mcp__repo__"


def test_namespaced_function_call_output_followup_with_previous_response_id(client):
    first = client.post(
        "/v1/responses",
        json={
            "input": "Use namespace tool please",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__repo__",
                    "description": "Repository tools.",
                    "tools": [
                        {
                            "type": "function",
                            "name": "list_files",
                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                        }
                    ],
                }
            ],
            "tool_choice": {"type": "function", "namespace": "mcp__repo__", "name": "list_files"},
            "store": True,
        },
    ).json()
    call = first["output"][0]

    second = client.post(
        "/v1/responses",
        json={
            "previous_response_id": first["id"],
            "input": [{"type": "function_call_output", "call_id": call["call_id"], "output": '["README.md"]'}],
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__repo__",
                    "description": "Repository tools.",
                    "tools": [
                        {
                            "type": "function",
                            "name": "list_files",
                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                        }
                    ],
                }
            ],
            "store": True,
        },
    )

    assert second.status_code == 200
    assert second.json()["output_text"] == 'Tool result: ["README.md"]'


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
    response = client.post(
        "/v1/responses",
        json={
            "input": "structured output failure fixture",
            "metadata": mock_metadata(structured_output="always_invalid"),
            "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
        },
    )
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
            "input": "function call loop guard",
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
