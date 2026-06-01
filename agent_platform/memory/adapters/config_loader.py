import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MEMORY_CONFIG: dict[str, Any] = {
    "hot_memory_max_chars": 2200,
    "user_memory_max_chars": 1375,
    "store_dir": "data/memory_dev",
    "skills_dir": "skills/published",
    "skills_drafts_dir": "skills/drafts",
    "external_profiles_dir": "data/external_profiles",
    "session_search_max_chars": 2000,
    "session_search_cache_ttl": 900,
    "retention_days": 90,
    "memory_summarizer_role": "memory_summarizer_llm",
    "use_llm_compress": True,
    "use_llm_summarize": True,
    "enable_session_vector_index": False,
    "session_vector_dir": "data/session_vectors",
    "session_vector_collection": "session_messages",
    "session_hybrid_search": True,
    "session_embedding_backend": "mock",
    "session_embedding_dim": 64,
    "archive_backend": "sqlite",
    "archive_sqlite_path": "data/session_archive.db",
    "archive_sqlite_pool_size": 5,
    "pg_host": "localhost",
    "pg_port": 5432,
    "pg_database": "agents",
    "pg_user": "postgres",
    "pg_password": "",
    "pg_pool_size": 10,
    "pg_dsn": None,
    "reindex_batch_size": 200,
    "enable_cold_archive": False,
    "cold_archive_prefix": "l2/cold",
    "cold_archive_compress": True,
}


def load_memory_config(config_path: str = "config/memory.yml") -> dict[str, Any]:
    env_path = os.environ.get("MEMORY_CONFIG")
    path = Path(env_path) if env_path else Path(config_path)
    cfg = dict(DEFAULT_MEMORY_CONFIG)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update(loaded)
    return cfg
