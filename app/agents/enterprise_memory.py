# -*- coding: utf-8 -*-
"""企业级记忆管理：会话列表、合规 purge、保留策略、配置摘要。"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from core.composition.run_context import RunContext
from core.domain.context import RequestContext

from agent_platform.memory.adapters.config_loader import load_memory_config

from app.agents.memory_views import list_pending_l1_deltas
from app.agents.memory_metrics import (
    get_memory_metric_stats,
    record_l1_confirm,
    record_purge,
)


def memory_config_summary(config_path: Optional[str] = None) -> Dict[str, Any]:
    """返回当前记忆配置摘要（不含密钥）。"""
    cfg = load_memory_config(config_path or "config/memory.yml")
    backend = str(cfg.get("archive_backend", "sqlite")).lower()
    return {
        "memory_config": os.environ.get("MEMORY_CONFIG") or "config/memory.yml",
        "archive_backend": backend,
        "l1_store_backend": str(cfg.get("l1_store_backend", "file")).lower(),
        "store_dir": cfg.get("store_dir"),
        "enable_session_vector_index": bool(cfg.get("enable_session_vector_index")),
        "enable_cold_archive": bool(cfg.get("enable_cold_archive")),
        "session_hybrid_search": bool(cfg.get("session_hybrid_search")),
        "retention_days": int(cfg.get("retention_days", 90)),
        "turn_buffer_flush_size": int(cfg.get("turn_buffer_flush_size", 0)),
        "cold_archive_encrypt_at_rest": bool(
            cfg.get("cold_archive_encrypt_at_rest")
        ),
        "external_profiles_backend": cfg.get("external_profiles_backend", "file"),
    }


async def list_user_sessions(
    ctx: RunContext,
    *,
    limit: int = 20,
) -> List[dict]:
    memory = ctx.require_memory()
    req = ctx.request
    lister = getattr(memory, "list_sessions", None)
    if lister is None:
        return []
    return await lister(req.tenant_id, req.user_id, limit=max(1, min(limit, 100)))


async def confirm_pending_l1(ctx: RunContext) -> int:
    memory = ctx.require_memory()
    confirmer = getattr(memory, "confirm_pending_deltas", None)
    if confirmer is None:
        return 0
    n = await confirmer(ctx.request)
    record_l1_confirm(ctx, n)
    return n


async def purge_user_memory(
    ctx: RunContext,
    *,
    tenant_id: str,
    user_id: str,
) -> dict:
    memory = ctx.require_memory()
    purger = getattr(memory, "purge_user_data", None)
    if purger is None:
        raise RuntimeError("MemoryPort 不支持 purge_user_data")
    req = RequestContext(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=ctx.request.session_id,
        trace_id=ctx.request.trace_id,
        channel=ctx.request.channel,
    )
    result = await purger(tenant_id, user_id)
    record_purge(ctx, scope="user")
    return result


async def run_retention_cleanup(
    ctx: RunContext,
    *,
    retention_days: Optional[int] = None,
) -> int:
    memory = ctx.require_memory()
    runner = getattr(memory, "purge_expired_sessions", None)
    if runner is None:
        raise RuntimeError("MemoryPort 不支持 purge_expired_sessions")
    cfg = load_memory_config()
    days = retention_days if retention_days is not None else int(
        cfg.get("retention_days", 90)
    )
    count = await runner(retention_days=days)
    record_purge(ctx, scope="retention", count=float(count))
    return count


async def purge_tenant_l3_memory(
    ctx: RunContext,
    *,
    tenant_id: str,
    delete_runs: bool = True,
) -> dict:
    memory = ctx.require_memory()
    purger = getattr(memory, "purge_tenant_l3", None)
    if purger is None:
        raise RuntimeError("MemoryPort 不支持 purge_tenant_l3")
    result = await purger(tenant_id, delete_runs=delete_runs)
    record_purge(ctx, scope="tenant_l3")
    return result


async def purge_tenant_l4_memory(
    ctx: RunContext,
    *,
    tenant_id: str,
) -> dict:
    memory = ctx.require_memory()
    purger = getattr(memory, "purge_tenant_l4", None)
    if purger is None:
        raise RuntimeError("MemoryPort 不支持 purge_tenant_l4")
    result = await purger(tenant_id)
    record_purge(ctx, scope="tenant_l4")
    return result


async def purge_old_checkpoints(
    ctx: RunContext,
    *,
    retention_days: Optional[int] = None,
) -> int:
    checkpointer = ctx.checkpointer
    if checkpointer is None:
        raise RuntimeError("RunContext 未注入 checkpointer")
    purger = getattr(checkpointer, "purge_older_than", None)
    if purger is None:
        raise RuntimeError("Checkpointer 不支持 purge_older_than")
    cfg = load_memory_config()
    days = retention_days if retention_days is not None else int(
        cfg.get("checkpoint_retention_days", cfg.get("retention_days", 90))
    )
    count = await purger(days)
    record_purge(ctx, scope="checkpoint", count=float(count))
    return count


async def get_memory_status(ctx: RunContext) -> dict:
    """租户/用户级记忆状态摘要。"""
    memory = ctx.require_memory()
    snap = memory.compose_prompt_snapshot(ctx.request)
    sessions = await list_user_sessions(ctx, limit=5)
    pending = list_pending_l1_deltas(ctx)
    return {
        "tenant_id": ctx.request.tenant_id,
        "user_id": ctx.request.user_id,
        "l1_hash": snap.hash,
        "l1_chars": len(snap.memory_text or ""),
        "pending_l1_count": len(pending),
        "recent_sessions": sessions,
        "config": memory_config_summary(),
        "metrics": get_memory_metric_stats(ctx),
    }
