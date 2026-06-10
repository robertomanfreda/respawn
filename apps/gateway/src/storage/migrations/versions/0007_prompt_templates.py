from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0007_prompt_templates"
down_revision: str | Sequence[str] | None = "0006_response_artifacts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "prompt_templates",
        sa.Column("record_id", sa.Text(), primary_key=True),
        sa.Column("prompt_id", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("template_json", json_type, nullable=False),
        sa.Column("metadata_json", json_type, nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_prompt_templates_prompt", "prompt_templates", ["prompt_id", "version"])
    op.create_index("ix_prompt_templates_tenant", "prompt_templates", ["tenant_id"])


def downgrade() -> None:
    op.drop_index("ix_prompt_templates_tenant", table_name="prompt_templates")
    op.drop_index("ix_prompt_templates_prompt", table_name="prompt_templates")
    op.drop_table("prompt_templates")
