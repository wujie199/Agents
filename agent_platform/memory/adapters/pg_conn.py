# -*- coding: utf-8 -*-
"""PostgreSQL 连接串构建（L2 + LangGraph Checkpointer 共用）。"""

from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import quote_plus


def build_postgres_conn_string(cfg: Optional[dict[str, Any]] = None) -> Optional[str]:
    """从 DATABASE_URL 或 memory 配置构建 postgres DSN。"""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url

    cfg = cfg or {}
    dsn = cfg.get("pg_dsn")
    if dsn:
        return str(dsn)

    host = cfg.get("pg_host") or os.environ.get("PGHOST", "localhost")
    port = int(cfg.get("pg_port") or os.environ.get("PGPORT", 5432))
    database = cfg.get("pg_database") or os.environ.get("PGDATABASE", "agents")
    user = cfg.get("pg_user") or os.environ.get("PGUSER", "postgres")
    password = cfg.get("pg_password") or os.environ.get("PGPASSWORD", "")

    if not database:
        return None

    user_q = quote_plus(str(user))
    if password:
        auth = f"{user_q}:{quote_plus(str(password))}"
    else:
        auth = user_q
    return f"postgresql://{auth}@{host}:{port}/{database}"


def is_postgresql_archive(cfg: dict[str, Any]) -> bool:
    return str(cfg.get("archive_backend", "sqlite")).lower() == "postgresql"
