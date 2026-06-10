from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0008_platform_files"
down_revision: str | Sequence[str] | None = "0007_prompt_templates"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "platform_files",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("purpose", sa.Text(), nullable=False),
        sa.Column("bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("mime_type", sa.Text(), nullable=True),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("storage_backend", sa.Text(), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=True),
        sa.Column("content_bytes", sa.LargeBinary(), nullable=True),
        sa.Column("metadata_json", json_type, nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_platform_files_tenant", "platform_files", ["tenant_id"])
    op.create_index("ix_platform_files_purpose", "platform_files", ["purpose"])
    op.create_index("ix_platform_files_expires", "platform_files", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_platform_files_expires", table_name="platform_files")
    op.drop_index("ix_platform_files_purpose", table_name="platform_files")
    op.drop_index("ix_platform_files_tenant", table_name="platform_files")
    op.drop_table("platform_files")
