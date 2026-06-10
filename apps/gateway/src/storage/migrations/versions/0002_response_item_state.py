from collections.abc import Sequence
import hashlib
from typing import Any

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0002_response_item_state"
down_revision: str | Sequence[str] | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
TOOL_ITEM_TYPES = {"function_call", "function_call_output", "tool_result"}


def upgrade() -> None:
    with op.batch_alter_table("response_items") as batch:
        batch.add_column(sa.Column("input_index", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("output_index", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("call_id", sa.Text(), nullable=True))
        batch.add_column(sa.Column("name", sa.Text(), nullable=True))
        batch.add_column(sa.Column("arguments_json", json_type, nullable=True))
        batch.add_column(sa.Column("output_json", json_type, nullable=True))
        batch.add_column(sa.Column("summary_json", json_type, nullable=True))
        batch.add_column(sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True))

    _backfill_response_items()

    with op.batch_alter_table("response_items") as batch:
        batch.create_unique_constraint("uq_response_items_input_index", ["response_id", "input_index"])
        batch.create_unique_constraint("uq_response_items_output_index", ["response_id", "output_index"])
        batch.create_unique_constraint("uq_response_items_call_id", ["response_id", "call_id"])
    op.create_index("ix_response_items_response_input", "response_items", ["response_id", "input_index"])
    op.create_index("ix_response_items_response_output", "response_items", ["response_id", "output_index"])


def downgrade() -> None:
    op.drop_index("ix_response_items_response_output", table_name="response_items")
    op.drop_index("ix_response_items_response_input", table_name="response_items")
    with op.batch_alter_table("response_items") as batch:
        batch.drop_constraint("uq_response_items_call_id", type_="unique")
        batch.drop_constraint("uq_response_items_output_index", type_="unique")
        batch.drop_constraint("uq_response_items_input_index", type_="unique")
        batch.drop_column("completed_at")
        batch.drop_column("summary_json")
        batch.drop_column("output_json")
        batch.drop_column("arguments_json")
        batch.drop_column("name")
        batch.drop_column("call_id")
        batch.drop_column("output_index")
        batch.drop_column("input_index")


def _backfill_response_items() -> None:
    bind = op.get_bind()
    responses = sa.table(
        "responses",
        sa.column("id", sa.Text()),
        sa.column("input_json", json_type),
        sa.column("output_json", json_type),
        sa.column("request_json", json_type),
        sa.column("status", sa.Text()),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("completed_at", sa.DateTime(timezone=True)),
    )
    response_items = sa.table(
        "response_items",
        sa.column("id", sa.Text()),
        sa.column("response_id", sa.Text()),
        sa.column("type", sa.Text()),
        sa.column("role", sa.Text()),
        sa.column("content_json", json_type),
        sa.column("status", sa.Text()),
        sa.column("input_index", sa.Integer()),
        sa.column("output_index", sa.Integer()),
        sa.column("call_id", sa.Text()),
        sa.column("name", sa.Text()),
        sa.column("arguments_json", json_type),
        sa.column("output_json", json_type),
        sa.column("summary_json", json_type),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("completed_at", sa.DateTime(timezone=True)),
    )

    for row in bind.execute(sa.select(responses)):
        response_id = row.id
        existing = bind.execute(
            sa.select(response_items.c.id).where(response_items.c.response_id == response_id)
        ).scalars()
        existing_ids = set(existing)
        for index, item in enumerate(_input_items_for_response(row)):
            item_id = item["id"]
            values = {
                "id": item_id,
                "response_id": response_id,
                "type": item.get("type", "message"),
                "role": item.get("role"),
                "content_json": item.get("content", []),
                "status": item.get("status", "completed"),
                "input_index": index,
                "output_index": None,
                "call_id": None,
                "name": None,
                "arguments_json": None,
                "output_json": None,
                "summary_json": item.get("summary"),
                "created_at": row.created_at,
                "completed_at": row.completed_at if item.get("status", "completed") in {"completed", "incomplete", "failed"} else None,
            }
            if item_id in existing_ids:
                bind.execute(response_items.update().where(response_items.c.id == item_id).values(**values))
            else:
                bind.execute(response_items.insert().values(**values))
            existing_ids.add(item_id)

        output_index = 0
        for item in _output_items_for_response(row):
            item_id = item["id"]
            values = {
                "id": item_id,
                "response_id": response_id,
                "type": item.get("type", "message"),
                "role": item.get("role"),
                "content_json": item.get("content", []),
                "status": item.get("status", _output_status(row.status)),
                "input_index": None,
                "output_index": output_index,
                "call_id": item.get("call_id"),
                "name": item.get("name"),
                "arguments_json": item.get("arguments"),
                "output_json": item.get("output"),
                "summary_json": item.get("summary"),
                "created_at": row.created_at,
                "completed_at": row.completed_at if item.get("status", _output_status(row.status)) in {"completed", "incomplete", "failed"} else None,
            }
            if item_id in existing_ids:
                bind.execute(response_items.update().where(response_items.c.id == item_id).values(**values))
            else:
                bind.execute(response_items.insert().values(**values))
            existing_ids.add(item_id)
            output_index += 1


def _input_items_for_response(row: Any) -> list[dict[str, Any]]:
    request_json = row.request_json if isinstance(row.request_json, dict) else {}
    input_value = request_json.get("input", row.input_json)
    if isinstance(input_value, str):
        return [
            {
                "id": _backfill_id("msg", row.id, "input", 0),
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": input_value}],
                "status": "completed",
            }
        ]
    if not isinstance(input_value, list):
        return []

    items: list[dict[str, Any]] = []
    for index, item in enumerate(input_value):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        role = item.get("role")
        if item_type in TOOL_ITEM_TYPES:
            continue
        if item_type == "reasoning":
            items.append(
                {
                    "id": str(item.get("id") or _backfill_id("rs", row.id, "input", index)),
                    "type": "reasoning",
                    "summary": item.get("summary", []),
                    "status": item.get("status", "completed"),
                }
            )
            continue
        if item_type == "message" or role in {"user", "assistant", "system", "developer"}:
            items.append(
                {
                    "id": str(item.get("id") or _backfill_id("msg", row.id, "input", index)),
                    "type": "message",
                    "role": role or "user",
                    "content": _input_content_parts(item.get("content", "")),
                    "status": item.get("status", "completed"),
                }
            )
    return items


def _output_items_for_response(row: Any) -> list[dict[str, Any]]:
    output = row.output_json
    if not isinstance(output, list):
        return []
    items = []
    for index, item in enumerate(output):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in TOOL_ITEM_TYPES:
            continue
        if item_type == "reasoning":
            items.append(
                {
                    "id": str(item.get("id") or _backfill_id("rs", row.id, "output", index)),
                    "type": "reasoning",
                    "summary": item.get("summary", []),
                    "status": item.get("status", _output_status(row.status)),
                }
            )
            continue
        if item_type in {None, "message"}:
            items.append(
                {
                    "id": str(item.get("id") or _backfill_id("msg", row.id, "output", index)),
                    "type": "message",
                    "role": item.get("role", "assistant"),
                    "content": item.get("content", []),
                    "status": item.get("status", _output_status(row.status)),
                }
            )
    return items


def _input_content_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text", part.get("output_text", ""))
            else:
                text = str(part)
            parts.append({"type": "input_text", "text": str(text)})
        return parts
    if isinstance(content, dict):
        return [{"type": "input_text", "text": str(content.get("text", content))}]
    return [{"type": "input_text", "text": str(content)}]


def _output_status(response_status: str) -> str:
    if response_status in {"completed", "incomplete", "failed"}:
        return response_status
    return "completed"


def _backfill_id(prefix: str, response_id: str, kind: str, index: int) -> str:
    digest = hashlib.sha256(f"{response_id}:{kind}:{index}".encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"
