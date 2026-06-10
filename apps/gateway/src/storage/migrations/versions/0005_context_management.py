from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0005_context_management"
down_revision: str | Sequence[str] | None = "0004_function_tool_protocol"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "response_context_events",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("response_id", sa.Text(), nullable=True),
        sa.Column("source_response_id", sa.Text(), nullable=True),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("strategy", sa.Text(), nullable=False),
        sa.Column("compacted_item_id", sa.Text(), nullable=True),
        sa.Column("source_item_ids_json", sa.JSON(), nullable=False),
        sa.Column("summary_json", sa.JSON(), nullable=True),
        sa.Column("input_tokens_before", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("input_tokens_after", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_response_context_events_response", "response_context_events", ["response_id"])


def downgrade() -> None:
    op.drop_index("ix_response_context_events_response", table_name="response_context_events")
    op.drop_table("response_context_events")
