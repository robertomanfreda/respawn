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

    assert {"responses", "response_items", "tool_calls", "usage_records"}.issubset(tables)
