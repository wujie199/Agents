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


async def list_user_sessions_enriched(
    ctx: RunContext,
    *,
    limit: int = 20,
) -> List[dict]:
    """在线 session + 冷归档 session（标注 storage）。"""
    cap = max(1, min(limit, 100))
    req = ctx.request
    memory = ctx.require_memory()
    rows = await list_user_sessions(ctx, limit=cap)
    by_id: dict[str, dict] = {}
    for row in rows:
        sid = str(row.get("session_id") or "")
        if not sid:
            continue
        enriched = dict(row)
        status = str(enriched.get("status") or "active")
        if status == "archived":
            enriched["storage"] = "cold"
        elif status == "closed":
            enriched["storage"] = "online_closed"
        else:
            enriched["storage"] = "online"
        by_id[sid] = enriched

    cold_lister = getattr(memory, "list_cold_archives", None)
    if cold_lister is not None:
        try:
            cold_rows = await cold_lister(req.tenant_id, req.user_id, limit=cap)
            for cold in cold_rows or []:
                sid = str(cold.get("session_id") or "")
                if not sid or sid in by_id:
                    continue
                by_id[sid] = {
                    "session_id": sid,
                    "user_id": req.user_id,
                    "tenant_id": req.tenant_id,
                    "status": "archived",
                    "storage": "cold",
                    "started_at": cold.get("started_at"),
                    "ended_at": cold.get("archived_at") or cold.get("ended_at"),
                    "message_count": cold.get("message_count"),
                }
        except Exception:
            pass

    merged = sorted(
        by_id.values(),
        key=lambda r: str(r.get("started_at") or r.get("ended_at") or ""),
        reverse=True,
    )
    return merged[:cap]


async def refresh_l4_profile(ctx: RunContext) -> dict:
    """会话中刷新 L4 外部画像（清缓存 + 重拉，不 merge L1）。"""
    memory = ctx.require_memory()
    refresher = getattr(memory, "refresh_external_profile", None)
    if refresher is None:
        raise RuntimeError("MemoryPort 不支持 refresh_external_profile")
    result = await refresher(ctx.request)
    if isinstance(getattr(ctx, "extra", None), dict):
        ctx.extra["l4_last_refresh"] = result
    return result


def format_finalize_summary(summary: dict | None) -> str:
    """格式化 finalize 结果供 REPL/API 展示。"""
    if not summary:
        return "（无 finalize 摘要）"
    parts: list[str] = []
    pending = int(summary.get("pending_applied") or 0)
    if pending:
        parts.append(f"pending L1 写入 {pending} 条")
    l1_ext = int(summary.get("l1_extract_pending") or 0)
    if l1_ext:
        parts.append(f"L2→L1 抽取 pending {l1_ext} 条")
    l4 = int(summary.get("l4_merged") or 0)
    if l4:
        keys = summary.get("l4_keys") or []
        key_txt = ", ".join(str(k) for k in keys if k) or f"{l4} 条"
        parts.append(f"L4→L1 合并 {l4} 条（{key_txt}）")
    if not parts:
        return "finalize 完成（无新增 L1 变更）"
    return "；".join(parts)


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
