import json
import sqlite3

from alembic import command
from alembic.config import Config


def test_alembic_upgrade_head_applies_initial_schema(tmp_path, monkeypatch):
    db_path = tmp_path / "migrations.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{db_path}")

    command.upgrade(config, "head")

    with sqlite3.connect(db_path) as connection:
        tables = {row[0] for row in connection.execute("select name from sqlite_master where type = 'table'")}
        response_item_columns = {row[1] for row in connection.execute("pragma table_info(response_items)")}
        background_job_columns = {row[1] for row in connection.execute("pragma table_info(background_jobs)")}

    assert {"responses", "response_items", "tool_calls", "usage_records", "background_jobs"}.issubset(tables)
    assert {
        "input_index",
        "output_index",
        "call_id",
        "name",
        "arguments_json",
        "output_json",
        "summary_json",
        "completed_at",
    }.issubset(response_item_columns)
    assert {
        "response_id",
        "status",
        "attempts",
        "timeout_at",
        "started_at",
        "heartbeat_at",
        "cancellation_requested_at",
        "completed_at",
        "error_json",
    }.issubset(background_job_columns)


def test_response_item_state_migration_backfills_supported_legacy_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy-migrations.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{db_path}")

    command.upgrade(config, "0001_initial")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            insert into responses
              (id, model, previous_response_id, status, input_json, output_json, request_json, metadata_json, usage_json, error_json, tenant_id, completed_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                "resp_legacy",
                "mock-model",
                None,
                "completed",
                json.dumps(
                    [
                        {"role": "user", "content": "legacy input"},
                        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "legacy reasoning"}]},
                        {"type": "function_call_output", "call_id": "call_legacy", "output": "{}"},
                    ]
                ),
                json.dumps(
                    [
                        {
                            "id": "msg_legacy_output",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "legacy output", "annotations": [], "logprobs": []}],
                        },
                        {"id": "call_legacy", "type": "function_call", "call_id": "call_legacy", "name": "legacy_tool", "arguments": "{}"},
                    ]
                ),
                json.dumps(
                    {
                        "input": [
                            {"role": "user", "content": "legacy input"},
                            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "legacy reasoning"}]},
                            {"type": "function_call_output", "call_id": "call_legacy", "output": "{}"},
                        ]
                    }
                ),
                json.dumps({}),
                json.dumps({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
                None,
                "tenant-legacy",
            ),
        )

    command.upgrade(config, "head")

    with sqlite3.connect(db_path) as connection:
        rows = list(
            connection.execute(
                """
                select id, type, role, input_index, output_index, content_json, summary_json
                from response_items
                where response_id = ?
                order by coalesce(input_index, 1000), coalesce(output_index, 1000), id
                """,
                ("resp_legacy",),
            )
        )

    assert len(rows) == 3
    assert rows[0][1:5] == ("message", "user", 0, None)
    assert json.loads(rows[0][5]) == [{"type": "input_text", "text": "legacy input"}]
    assert rows[1][1:5] == ("reasoning", None, 1, None)
    assert json.loads(rows[1][6]) == [{"type": "summary_text", "text": "legacy reasoning"}]
    assert rows[2][0] == "msg_legacy_output"
    assert rows[2][1:5] == ("message", "assistant", None, 0)
