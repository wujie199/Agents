# -*- coding: utf-8 -*-
"""企业级依赖健康检查：Redis / L2 Archive / 冷归档 / Checkpointer。"""

from __future__ import annotations

import os
from typing import Any, Literal, Optional

from agent_platform.memory.adapters.config_loader import load_memory_config
from agent_platform.memory.adapters.archive_factory import build_archive_db
from core.composition.production_factory import build_cache_port

ChatProfile = Literal["dev", "production"]


async def _check_redis() -> dict[str, Any]:
    prefix = os.environ.get("CHAT_CACHE_REDIS_PREFIX", "agents")
    cache = build_cache_port(pool_size=2, prefix=prefix)
    if hasattr(cache, "health"):
        result = await cache.health()
        if isinstance(result, dict):
            result.setdefault("prefix", prefix)
            return result
    return {"status": "unknown", "type": "redis"}


async def _check_archive(
    config_dir: str,
    data_dir: str,
    *,
    profile: ChatProfile,
) -> dict[str, Any]:
    mem_cfg = load_memory_config(f"{config_dir}/memory.yml")
    db_name = "dev_archive.db" if profile == "dev" else None
    archive = build_archive_db(
        mem_cfg,
        data_dir=data_dir,
        db_name=db_name or "session_archive.db",
    )
    try:
        if hasattr(archive, "_init_pool"):
            await archive._init_pool()
        if hasattr(archive, "health_check"):
            health = await archive.health_check()
        elif hasattr(archive, "health"):
            raw = archive.health()
            health = await raw if hasattr(raw, "__await__") else raw
        else:
            health = {"status": "unknown"}
        backend = mem_cfg.get("archive_backend", "sqlite")
        out: dict[str, Any] = {
            "status": health.get("status", "unknown"),
            "backend": backend,
            "detail": health,
        }
        if backend == "postgresql" and hasattr(archive, "_get_connection"):
            try:
                async with archive._get_connection() as conn:
                    for table in ("sessions", "messages", "tool_calls"):
                        count = await conn.fetchval(
                            f"SELECT COUNT(*) FROM {table}"
                        )
                        out.setdefault("counts", {})[table] = int(count or 0)
            except Exception as e:
                out["counts_error"] = str(e)
        return out
    finally:
        if hasattr(archive, "close"):
            close = archive.close()
            if hasattr(close, "__await__"):
                await close


async def _check_object_store(
    config_dir: str,
    data_dir: str,
) -> dict[str, Any]:
    mem_cfg = load_memory_config()
    if not mem_cfg.get("enable_cold_archive"):
        return {"status": "skipped", "reason": "enable_cold_archive=false"}
    from agent_platform.storage.adapters.s3.s3_object_store_adapter import (
        S3ObjectStoreAdapter,
    )

    store = S3ObjectStoreAdapter(
        endpoint_url=os.environ.get("S3_ENDPOINT"),
        access_key=os.environ.get("S3_ACCESS_KEY", ""),
        secret_key=os.environ.get("S3_SECRET_KEY", ""),
        bucket_name=os.environ.get("S3_BUCKET", "agents-storage"),
        local_fallback_dir=f"{data_dir}/objects",
    )
    if hasattr(store, "health"):
        raw = store.health()
        return await raw if hasattr(raw, "__await__") else raw
    return {"status": "unknown"}


async def _check_checkpointer(
    config_dir: str,
    data_dir: str,
    *,
    profile: ChatProfile,
) -> dict[str, Any]:
    mem_cfg = load_memory_config(f"{config_dir}/memory.yml")
    db_name = "dev_archive.db" if profile == "dev" else None
    archive = build_archive_db(
        mem_cfg,
        data_dir=data_dir,
        db_name=db_name or "session_archive.db",
    )
    try:
        if hasattr(archive, "_init_pool"):
            await archive._init_pool()
        from agent_platform.memory.adapters.relational_checkpointer_adapter import (
            RelationalCheckpointerAdapter,
        )

        cp = RelationalCheckpointerAdapter(archive)
        if hasattr(cp, "health_check"):
            return await cp.health_check()
        return {"status": "unknown"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
    finally:
        if hasattr(archive, "close"):
            close = archive.close()
            if hasattr(close, "__await__"):
                await close


def _aggregate_status(
    checks: dict[str, dict[str, Any]],
    *,
    profile: ChatProfile,
    strict: bool,
) -> str:
    required = ["redis", "archive"]
    if profile == "production":
        if strict:
            required.append("object_store")
    for name in required:
        st = (checks.get(name) or {}).get("status")
        if st not in ("healthy", "skipped"):
            return "unhealthy"
    return "healthy"


async def run_health_checks(
    *,
    config_dir: str = "config",
    data_dir: str = "data",
    profile: ChatProfile = "dev",
    strict: bool = False,
) -> dict[str, Any]:
    """聚合依赖探测；strict=True 时 production 要求冷归档存储可用。"""
    checks: dict[str, dict[str, Any]] = {}
    checks["redis"] = await _check_redis()
    checks["archive"] = await _check_archive(
        config_dir, data_dir, profile=profile
    )
    checks["object_store"] = await _check_object_store(config_dir, data_dir)
    checks["checkpointer"] = await _check_checkpointer(
        config_dir, data_dir, profile=profile
    )
    status = _aggregate_status(checks, profile=profile, strict=strict)
    return {
        "status": status,
        "profile": profile,
        "checks": checks,
    }
