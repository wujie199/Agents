"""PostgreSQL 连接串构建单元测试。"""

from __future__ import annotations

import os
from unittest.mock import patch

from agent_platform.memory.adapters.pg_conn import (
    build_postgres_conn_string,
    is_postgresql_archive,
)


def test_is_postgresql_archive():
    assert is_postgresql_archive({"archive_backend": "postgresql"})
    assert not is_postgresql_archive({"archive_backend": "sqlite"})


def test_build_from_pg_dsn():
    conn = build_postgres_conn_string({"pg_dsn": "postgresql://u:p@db:5432/agents"})
    assert conn == "postgresql://u:p@db:5432/agents"


def test_build_from_fields():
    conn = build_postgres_conn_string(
        {
            "pg_host": "localhost",
            "pg_port": 5433,
            "pg_database": "agents",
            "pg_user": "app",
            "pg_password": "s@cret",
        }
    )
    assert conn.startswith("postgresql://")
    assert "localhost:5433/agents" in conn


def test_build_prefers_database_url():
    with patch.dict(
        os.environ,
        {"DATABASE_URL": "postgresql://env@host/db"},
        clear=False,
    ):
        conn = build_postgres_conn_string({"pg_dsn": "postgresql://ignored/db"})
    assert conn == "postgresql://env@host/db"
