"""L4 ExternalMemoryProvider 工厂：file / http / noop + 可选 TTL 缓存。"""

from __future__ import annotations

import os
from typing import Any

from agent_platform.memory.adapters.cached_external_memory_adapter import (
    CachedExternalMemoryAdapter,
)
from agent_platform.memory.adapters.file_external_memory_adapter import (
    FileExternalMemoryAdapter,
)
from agent_platform.memory.adapters.http_external_memory_adapter import (
    HttpExternalMemoryAdapter,
)
from agent_platform.memory.adapters.noop_external_adapter import (
    NoOpExternalMemoryAdapter,
)
def build_external_memory(cfg: dict[str, Any]) -> Any:
    backend = str(cfg.get("external_profiles_backend", "file")).lower()
    if backend == "noop":
        inner = NoOpExternalMemoryAdapter()
    elif backend == "http":
        base_url = cfg.get("external_profiles_http_url") or ""
        if not base_url:
            raise ValueError(
                "external_profiles_backend=http requires external_profiles_http_url"
            )
        api_key = cfg.get("external_profiles_http_api_key") or os.environ.get(
            "EXTERNAL_PROFILES_API_KEY"
        )
        inner = HttpExternalMemoryAdapter(
            base_url,
            timeout=float(cfg.get("external_profiles_http_timeout", 10)),
            api_key=api_key,
        )
    else:
        inner = FileExternalMemoryAdapter(
            profiles_dir=cfg.get(
                "external_profiles_dir", "data/external_profiles"
            )
        )

    ttl = int(cfg.get("external_profile_cache_ttl", 0))
    if ttl > 0:
        cache_backend = str(
            cfg.get("external_profile_cache_backend", "redis")
        ).lower()
        cache_port = None
        if cache_backend == "redis":
            from core.composition.production_factory import build_cache_port

            cache_port = build_cache_port(prefix="l4", pool_size=5)
        return CachedExternalMemoryAdapter(
            inner, ttl_seconds=ttl, cache_port=cache_port
        )
    return inner
