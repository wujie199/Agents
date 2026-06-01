from agent_platform.memory.adapters.archive_factory import build_archive_db
from agent_platform.memory.adapters.config_loader import DEFAULT_MEMORY_CONFIG
from agent_platform.storage.adapters.sqlite.relational_adapter import (
    AsyncSQLiteRelationalAdapter,
)


def test_build_archive_db_sqlite_default(tmp_path):
    cfg = dict(DEFAULT_MEMORY_CONFIG)
    db = build_archive_db(cfg, data_dir=str(tmp_path))
    assert isinstance(db, AsyncSQLiteRelationalAdapter)


def test_build_archive_db_postgresql_type():
    cfg = {**DEFAULT_MEMORY_CONFIG, "archive_backend": "postgresql"}
    from agent_platform.storage.adapters.postgresql.relational_adapter import (
        PostgreSQLAdapter,
    )

    db = build_archive_db(cfg, data_dir="/tmp")
    assert isinstance(db, PostgreSQLAdapter)


def test_pg_build_tsquery_terms():
    from agent_platform.storage.adapters.postgresql.relational_adapter import (
        PostgreSQLAdapter,
    )

    assert PostgreSQLAdapter._build_tsquery_terms("hello world") == "hello | world"
    assert PostgreSQLAdapter._build_tsquery_terms("  ") == ""


def test_parse_database_url():
    from agent_platform.memory.adapters.archive_factory import parse_database_url

    parsed = parse_database_url("postgresql://app:secret@db.example.com:5433/agents")
    assert parsed["pg_host"] == "db.example.com"
    assert parsed["pg_port"] == 5433
    assert parsed["pg_database"] == "agents"
    assert parsed["pg_user"] == "app"
    assert parsed["pg_password"] == "secret"
