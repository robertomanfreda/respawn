from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0001_initial"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "responses",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("previous_response_id", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("input_json", json_type, nullable=False),
        sa.Column("output_json", json_type, nullable=False),
        sa.Column("request_json", json_type, nullable=False),
        sa.Column("metadata_json", json_type, nullable=False, server_default=sa.text("'{}'")),
        sa.Column("usage_json", json_type, nullable=True),
        sa.Column("error_json", json_type, nullable=True),
        sa.Column("tenant_id", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "response_items",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("response_id", sa.Text(), sa.ForeignKey("responses.id"), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=True),
        sa.Column("content_json", json_type, nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "tool_calls",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("response_id", sa.Text(), sa.ForeignKey("responses.id"), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("arguments_json", json_type, nullable=False),
        sa.Column("output_json", json_type, nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "usage_records",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("response_id", sa.Text(), sa.ForeignKey("responses.id"), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("usage_records")
    op.drop_table("tool_calls")
    op.drop_table("response_items")
    op.drop_table("responses")
