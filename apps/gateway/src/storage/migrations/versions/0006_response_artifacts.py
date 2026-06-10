from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0006_response_artifacts"
down_revision: str | Sequence[str] | None = "0005_context_management"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "response_artifacts",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("response_id", sa.Text(), sa.ForeignKey("responses.id"), nullable=False),
        sa.Column("item_id", sa.Text(), nullable=False),
        sa.Column("content_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=True),
        sa.Column("mime_type", sa.Text(), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_json", json_type, nullable=False),
        sa.Column("metadata_json", json_type, nullable=False),
        sa.Column("content_json", json_type, nullable=True),
        sa.Column("tenant_id", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_response_artifacts_response", "response_artifacts", ["response_id"])
    op.create_index("ix_response_artifacts_item", "response_artifacts", ["response_id", "item_id"])
    op.create_index("ix_response_artifacts_tenant", "response_artifacts", ["tenant_id"])


def downgrade() -> None:
    op.drop_index("ix_response_artifacts_tenant", table_name="response_artifacts")
    op.drop_index("ix_response_artifacts_item", table_name="response_artifacts")
    op.drop_index("ix_response_artifacts_response", table_name="response_artifacts")
    op.drop_table("response_artifacts")
