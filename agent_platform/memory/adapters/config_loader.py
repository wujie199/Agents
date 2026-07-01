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


def resolve_memory_config_path(
    config_dir: str = "config",
    *,
    profile: str = "dev",
) -> str:
    """解析记忆配置文件路径（尊重 MEMORY_CONFIG / production 默认）。"""
    env_path = os.environ.get("MEMORY_CONFIG")
    if env_path:
        return env_path
    base = Path(config_dir)
    if profile == "production":
        for name in ("memory.production.yml", "memory.production.example.yml"):
            candidate = base / name
            if candidate.is_file():
                return str(candidate)
    return str(base / "memory.yml")


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
    cfg = dict(DEFAULT_MEMORY_CONFIG)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update(loaded)
    cfg["_config_path"] = str(path)
    return cfg
