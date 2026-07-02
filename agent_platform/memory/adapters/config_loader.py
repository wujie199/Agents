import os
from pathlib import Path
from typing import Any

from agent_platform.memory.adapters.memory_yaml import (
    load_memory_yaml_document,
    resolve_memory_config_path,
)

_LEGACY_MEMORY_FILENAMES = frozenset(
    {
        "memory.dev-vector.example.yml",
        "memory.dev-cold.example.yml",
        "memory.dev-skills.example.yml",
        "memory.dev-l4-http.example.yml",
    }
)

DEFAULT_MEMORY_CONFIG: dict[str, Any] = {
    "hot_memory_max_chars": 2200,
    "user_memory_max_chars": 1375,
    "store_dir": "data/memory_dev",
    "skills_dir": "skills/published",
    "skills_drafts_dir": "skills/drafts",
    "external_profiles_dir": "data/external_profiles",
    "session_search_max_chars": 2000,
    "session_search_cache_ttl": 900,
    "session_search_negative_cache_ttl": 120,
    "retention_days": 90,
    "checkpoint_retention_days": 90,
    "memory_summarizer_role": "memory_summarizer_llm",
    "use_llm_compress": True,
    "use_llm_summarize": True,
    "enable_session_vector_index": True,
    "session_vector_dir": "data/session_vectors",
    "session_vector_collection": "session_messages",
    "session_hybrid_search": True,
    "session_vector_embed_batch_size": 32,
    "session_vector_auto_reindex_on_version_change": True,
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
    "enable_cold_archive": True,
    "cold_archive_prefix": "l2/cold",
    "cold_archive_compress": True,
    "cold_archive_search_scan_limit": 100,
    "cold_archive_keep_vectors": True,
    "cold_archive_encrypt_at_rest": False,
    "session_search_cold_fallback": True,
    "session_search_rerank": True,
    "turn_buffer_flush_size": 10,
    "l1_use_file_lock": True,
    "l1_store_backend": "file",
    "l1_hermes_entries": True,
    "l1_write_approval": False,
    "l1_nudge_interval": 10,
    "l2_compression_continuation": "split",
    "skills_meta_dir": "skills/meta",
    "skill_auto_extract_draft": True,
    "skill_deprecate_threshold": 0.2,
    "skill_include_deprecated_in_search": False,
    "skill_auto_extract_min_steps": 2,
    "external_profiles_backend": "file",
    "external_profiles_http_url": None,
    "external_profiles_http_timeout": 10,
    "external_profiles_http_api_key": None,
    "external_profile_cache_ttl": 300,
    "external_profile_cache_backend": "redis",
    "external_merge_on_finalize": True,
    "purge_delete_external_audit": True,
    "purge_tenant_l4_strip_user_keys": True,
}


def ensure_memory_config_env(
    config_dir: str = "config",
    *,
    profile: str = "dev",
) -> str:
    """production profile 未设 MEMORY_CONFIG 时自动指向生产示例配置。"""
    if os.environ.get("MEMORY_CONFIG"):
        return os.environ["MEMORY_CONFIG"]
    path = resolve_memory_config_path(config_dir, profile=profile)
    if profile == "production":
        os.environ.setdefault("MEMORY_CONFIG", path)
    return path


def load_memory_config(config_path: str = "config/memory.yml") -> dict[str, Any]:
    env_path = os.environ.get("MEMORY_CONFIG")
    path = Path(env_path) if env_path else Path(config_path)
    if (
        env_path
        and not path.is_file()
        and path.name not in _LEGACY_MEMORY_FILENAMES
    ):
        path = Path(config_path)
    config_dir = str(path.parent) if path.parent.is_dir() else "config"

    cfg = dict(DEFAULT_MEMORY_CONFIG)
    loaded = load_memory_yaml_document(path, config_dir=config_dir)
    if loaded or path.exists():
        cfg.update(loaded)
    cfg["_config_path"] = str(path if path.exists() or env_path else config_path)
    return cfg
