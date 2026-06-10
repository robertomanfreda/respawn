from collections.abc import Sequence

from alembic import op


revision: str = "0004_function_tool_protocol"
down_revision: str | Sequence[str] | None = "0003_background_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("response_items") as batch:
        batch.drop_constraint("uq_response_items_call_id", type_="unique")
    op.create_index("ix_response_items_response_call", "response_items", ["response_id", "call_id"])


def downgrade() -> None:
    op.drop_index("ix_response_items_response_call", table_name="response_items")
    with op.batch_alter_table("response_items") as batch:
        batch.create_unique_constraint("uq_response_items_call_id", ["response_id", "call_id"])
