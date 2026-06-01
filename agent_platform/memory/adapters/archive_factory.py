import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse


def parse_database_url(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    if parsed.scheme not in ("postgresql", "postgres"):
        raise ValueError(f"Unsupported DATABASE_URL scheme: {parsed.scheme!r}")
    database = (parsed.path or "").lstrip("/")
    if not database:
        raise ValueError("DATABASE_URL must include database name")
    return {
        "pg_host": parsed.hostname or "localhost",
        "pg_port": parsed.port or 5432,
        "pg_database": database,
        "pg_user": parsed.username or "postgres",
        "pg_password": parsed.password or "",
    }


def build_archive_db(
    cfg: dict[str, Any],
    *,
    data_dir: str,
    db_name: str = "session_archive.db",
    force_backend: Optional[str] = None,
) -> Any:
    backend = force_backend or str(cfg.get("archive_backend", "sqlite")).lower()

    if backend == "postgresql":
        from agent_platform.storage.adapters.postgresql.relational_adapter import (
            PostgreSQLAdapter,
        )

        pg_cfg = dict(cfg)
        database_url = os.environ.get("DATABASE_URL") or cfg.get("pg_dsn")
        if database_url:
            pg_cfg.update(parse_database_url(database_url))

        return PostgreSQLAdapter(
            host=pg_cfg.get("pg_host") or os.environ.get("PGHOST", "localhost"),
            port=int(pg_cfg.get("pg_port") or os.environ.get("PGPORT", 5432)),
            database=pg_cfg.get("pg_database")
            or os.environ.get("PGDATABASE", "agents"),
            user=pg_cfg.get("pg_user") or os.environ.get("PGUSER", "postgres"),
            password=pg_cfg.get("pg_password")
            or os.environ.get("PGPASSWORD", ""),
            pool_size=int(pg_cfg.get("pg_pool_size", 10)),
        )

    from agent_platform.storage.adapters.sqlite.relational_adapter import (
        AsyncSQLiteRelationalAdapter,
    )

    db_path = cfg.get("archive_sqlite_path") or str(Path(data_dir) / db_name)
    return AsyncSQLiteRelationalAdapter(
        db_path=db_path,
        pool_size=int(cfg.get("archive_sqlite_pool_size", 5)),
    )
