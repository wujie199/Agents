# -*- coding: utf-8 -*-
"""记忆子系统运行状态采集与 NDJSON 调试日志。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from core.composition.run_context import RunContext

from agent_platform.memory.adapters.config_loader import load_memory_config

from app.agents.memory.enterprise_memory import memory_config_summary
from app.agents.memory.memory_metrics import get_memory_metric_stats
from app.agents.memory.memory_views import list_pending_l1_deltas

_DEBUG_LOG_DEFAULT = str(
    Path(__file__).resolve().parents[2] / ".cursor" / "memory_runtime.ndjson"
)
_SESSION_ID = os.environ.get("MEMORY_DEBUG_SESSION", "default")
_ENABLED = os.environ.get("MEMORY_RUNTIME_DEBUG", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_VERBOSE = False


def debug_log_path() -> Path:
    return Path(os.environ.get("MEMORY_DEBUG_LOG", _DEBUG_LOG_DEFAULT))


def set_memory_runtime_debug(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = enabled


def set_memory_runtime_verbose(enabled: bool) -> None:
    global _VERBOSE
    _VERBOSE = enabled


def is_memory_runtime_debug() -> bool:
    return _ENABLED


def is_memory_runtime_verbose() -> bool:
    return _VERBOSE


def resolve_memory_trace(
    *,
    profile: str = "dev",
    debug: bool = False,
    debug_quiet: bool = False,
    no_debug: bool = False,
) -> tuple[bool, bool]:
    """解析是否开启 trace 及是否打印控制台 debug。返回 (trace_on, console_debug)。"""
    env = os.environ.get("MEMORY_RUNTIME_DEBUG", "").lower()
    agent_env = os.environ.get("AGENT_DEBUG", "").lower()
    if no_debug or env in ("0", "false", "no", "off"):
        return False, False
    if env in ("1", "true", "yes", "on"):
        return True, debug and not debug_quiet
    if agent_env in ("1", "true", "yes", "on"):
        return True, debug and not debug_quiet
    if debug or debug_quiet:
        return True, debug and not debug_quiet
    return False, False


def chat_config_verify_snapshot(cc: Any) -> dict[str, Any]:
    """Round2 验证：startup/turn 日志中可对照的关键 chat 配置。"""
    return {
        "enable_rag_gating": cc.enable_rag_gating,
        "rag_gating_min_chars": cc.rag_gating_min_chars,
        "max_history_turns": cc.max_history_turns,
        "max_history_chars": cc.max_history_chars,
        "enable_memory_tools": cc.enable_memory_tools,
        "remember_require_hitl": cc.remember_require_hitl,
        "auto_confirm_pending_on_exit": cc.auto_confirm_pending_on_exit,
        "enable_l1_extract_on_finalize": cc.enable_l1_extract_on_finalize,
        "l1_extract_allowed_keys": list(cc.l1_extract_allowed_keys),
        "interactive_flush_buffer": cc.interactive_flush_buffer,
        "retrieval_orchestration": cc.retrieval_orchestration,
        "knowledge_session_search": cc.knowledge_session_search,
        "l4_prefetch_on_knowledge": cc.l4_prefetch_on_knowledge,
        "rolling_summary_user_only": cc.rolling_summary_user_only,
        "evidence_strict_grounding": cc.evidence_strict_grounding,
        "recall_tool_hint_on_miss": cc.recall_tool_hint_on_miss,
        "recall_skip_rag_when_prefetch_miss": cc.recall_skip_rag_when_prefetch_miss,
        "retrieval_llm_router": cc.retrieval_llm_router,
    }


def trace_write(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
    run_id: str = "default",
    force: bool = False,
) -> None:
    if not _ENABLED and not force:
        return
    payload = {
        "sessionId": _SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        log_path = debug_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass


_write_ndjson = trace_write  # 内部别名


def _perf_run_id(ctx: RunContext | None) -> str:
    if ctx is not None and getattr(ctx, "request", None) is not None:
        return getattr(ctx.request, "session_id", None) or "default"
    return "default"


def perf_begin(ctx: RunContext | None, label: str = "turn") -> None:
    """开始一轮耗时统计（不重复重置已有 turn_perf）。"""
    if ctx is None or not isinstance(getattr(ctx, "extra", None), dict):
        return
    if "turn_perf" in ctx.extra:
        return
    ctx.extra["turn_perf"] = {
        "label": label,
        "start": time.perf_counter(),
        "stages": [],
    }


def perf_mark(
    ctx: RunContext | None,
    stage: str,
    duration_ms: float,
    **extra: Any,
) -> None:
    """记录单阶段耗时（写入 NDJSON + 累积到 ctx.extra.turn_perf）。"""
    ms = round(duration_ms, 2)
    entry = {"stage": stage, "duration_ms": ms, **extra}
    if ctx is not None and isinstance(getattr(ctx, "extra", None), dict):
        perf = ctx.extra.get("turn_perf")
        if isinstance(perf, dict):
            perf.setdefault("stages", []).append(entry)
    trace_write(
        hypothesis_id="PERF-LATENCY",
        location=f"perf:{stage}",
        message="stage timing",
        data=entry,
        run_id=_perf_run_id(ctx),
    )


async def perf_await(ctx: RunContext | None, stage: str, coro: Any, **extra: Any) -> Any:
    """await 协程并记录耗时。"""
    t0 = time.perf_counter()
    try:
        return await coro
    finally:
        perf_mark(ctx, stage, (time.perf_counter() - t0) * 1000, **extra)


def perf_sync(
    ctx: RunContext | None,
    stage: str,
    fn: Any,
    *args: Any,
    **extra: Any,
) -> Any:
    """同步调用并记录耗时。"""
    t0 = time.perf_counter()
    try:
        return fn(*args)
    finally:
        perf_mark(ctx, stage, (time.perf_counter() - t0) * 1000, **extra)


_PERF_ROLLUP_STAGES = frozenset(
    {
        "execute_chat_turn_wall",
        "stream_chat_turn_events_wall",
        "GRAPH.langgraph_ainvoke",
        "GRAPH.prepare_node",
        "RAG.retrieve_bundle",
        "CTX.prepare_session_context",
        "CTX.prefetch_parallel",
        "LLM.stream_wall",
    }
)


def perf_finish_turn(
    ctx: RunContext | None,
    *,
    phase: str = "turn_done",
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """输出本轮耗时汇总（按阶段排序）。"""
    if ctx is None or not isinstance(getattr(ctx, "extra", None), dict):
        return
    perf = ctx.extra.get("turn_perf")
    if not isinstance(perf, dict):
        return
    total_ms = round((time.perf_counter() - perf.get("start", time.perf_counter())) * 1000, 2)
    stages = list(perf.get("stages") or [])
    leaf_stages = [
        s for s in stages if s.get("stage") not in _PERF_ROLLUP_STAGES
    ]
    ranked = sorted(leaf_stages, key=lambda s: s.get("duration_ms", 0), reverse=True)
    accounted_ms = round(sum(s.get("duration_ms", 0) for s in leaf_stages), 2)
    trace_write(
        hypothesis_id="PERF-SUMMARY",
        location=f"perf.finish:{phase}",
        message="turn latency summary",
        data={
            "phase": phase,
            "label": perf.get("label"),
            "total_ms": total_ms,
            "accounted_ms": accounted_ms,
            "unaccounted_ms": round(max(0.0, total_ms - accounted_ms), 2),
            "stage_count": len(leaf_stages),
            "rollup_stage_count": len(stages) - len(leaf_stages),
            "slowest": ranked[:8],
            "all_stages": stages,
            "leaf_stages": leaf_stages,
            **(extra or {}),
        },
        run_id=_perf_run_id(ctx),
    )
    try:
        from app.agents.memory.memory_metrics import record_turn_latency

        record_turn_latency(ctx, total_ms)
    except Exception:
        pass
    ctx.extra.pop("turn_perf", None)


def clear_layer_triggers(ctx: RunContext | None) -> None:
    if ctx is not None and isinstance(getattr(ctx, "extra", None), dict):
        ctx.extra["layer_triggers"] = []


def trace_layer_trigger(
    ctx: RunContext | None,
    layer: str,
    action: str,
    triggered: bool,
    reason: str = "",
    *,
    data: Optional[dict[str, Any]] = None,
    run_id: str = "default",
) -> None:
    """记录单层记忆/RAG 是否触发（写入 NDJSON + 累积到 ctx.extra）。"""
    from app.agents.debug.agent_pipeline_debug import pipeline_debug_enabled, pipeline_debug_log

    entry = {
        "layer": layer,
        "action": action,
        "triggered": triggered,
        "reason": reason,
        "data": data or {},
        "ts_ms": int(time.time() * 1000),
    }
    if ctx is not None and isinstance(getattr(ctx, "extra", None), dict):
        triggers = ctx.extra.setdefault("layer_triggers", [])
        triggers.append(entry)
        run_id = getattr(ctx.request, "session_id", run_id)
    if pipeline_debug_enabled():
        pipeline_debug_log(
            component=layer,
            stage=action,
            status="ON" if triggered else "SKIP",
            location=f"layer_trace.{layer}.{action}",
            message=reason or ("triggered" if triggered else "skipped"),
            data={"triggered": triggered, "reason": reason, **(data or {})},
            run_id=run_id,
            hypothesis_id=f"TRIGGER-{layer}",
        )
    if not _ENABLED:
        return
    trace_write(
        hypothesis_id=f"TRIGGER-{layer}",
        location=f"layer_trace.{layer}.{action}",
        message=f"{'ON' if triggered else 'SKIP'} {layer} {action}",
        data={
            "triggered": triggered,
            "reason": reason,
            **(data or {}),
        },
        run_id=run_id,
    )


def format_layer_triggers(ctx: RunContext | None) -> str:
    triggers = []
    if ctx is not None:
        triggers = list((ctx.extra or {}).get("layer_triggers") or [])
    if not triggers:
        return "（本轮暂无 layer_triggers，请先发送一条消息）"
    lines = ["── 本轮 L1-L4 / RAG 触发链 ──"]
    for i, t in enumerate(triggers, 1):
        flag = "✓" if t.get("triggered") else "✗"
        reason = t.get("reason") or ""
        lines.append(
            f"  {i}. [{flag}] {t.get('layer')}::{t.get('action')}  {reason}"
        )
        extra = t.get("data") or {}
        if extra:
            preview = json.dumps(extra, ensure_ascii=False)
            if len(preview) > 120:
                preview = preview[:120] + "…"
            lines.append(f"       {preview}")
    lines.append(f"  log → {debug_log_path()}")
    if ctx is not None:
        perf = (ctx.extra or {}).get("turn_perf") or {}
        stages = list(perf.get("stages") or [])
        if stages:
            ranked = sorted(
                stages,
                key=lambda s: float(s.get("duration_ms") or 0),
                reverse=True,
            )[:3]
            lines.append("── 耗时 Top3（ms）──")
            for row in ranked:
                lines.append(
                    f"  · {row.get('stage')}: {row.get('duration_ms')}ms"
                )
        decision = (ctx.extra or {}).get("turn_decision")
        if isinstance(decision, dict):
            lines.append("── 本轮决策 ──")
            lines.append(
                f"  intent={decision.get('intent')} "
                f"rag={decision.get('run_rag')} "
                f"recall_hit={decision.get('recall_prefetch_hit')}"
            )
            if decision.get("skip_rag_reason"):
                lines.append(f"  skip_rag={decision.get('skip_rag_reason')}")
    return "\n".join(lines)


def log_layer_trigger_summary(
    ctx: RunContext,
    *,
    user_message: str = "",
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """一轮 build_turn 结束：汇总触发链写入 NDJSON。"""
    from app.agents.debug.agent_pipeline_debug import (
        log_turn_pipeline_summary,
        pipeline_debug_enabled,
    )

    if pipeline_debug_enabled():
        log_turn_pipeline_summary(ctx, user_message=user_message, extra=extra)
    if not _ENABLED:
        return
    triggers = list((ctx.extra or {}).get("layer_triggers") or [])
    summary = {
        "user_preview": _preview(user_message, 120),
        "trigger_count": len(triggers),
        "triggered_layers": [
            f"{t.get('layer')}:{t.get('action')}"
            for t in triggers
            if t.get("triggered")
        ],
        "skipped_layers": [
            f"{t.get('layer')}:{t.get('action')}({t.get('reason')})"
            for t in triggers
            if not t.get("triggered")
        ],
        "timeline": triggers,
    }
    if extra:
        summary.update(extra)
    trace_write(
        hypothesis_id="TRIGGER-SUMMARY",
        location="layer_trace.summary",
        message="本轮记忆/RAG 触发汇总",
        data=summary,
        run_id=ctx.request.session_id,
    )


def _preview(text: str | None, limit: int = 300) -> str:
    s = (text or "").replace("\n", "\\n")
    return s[:limit] + ("…" if len(s) > limit else "")


def _fact_row(fact: Any) -> dict[str, Any]:
    if isinstance(fact, dict):
        return {
            "key": fact.get("key"),
            "value": fact.get("value"),
            "source": fact.get("source"),
        }
    return {
        "key": getattr(fact, "key", None),
        "value": getattr(fact, "value", None),
        "source": getattr(fact, "source", None),
    }


def memory_debug_console_enabled() -> bool:
    """trace 开启时默认打印轮次摘要；MEMORY_DEBUG_CONSOLE=0 可关闭。"""
    if not _ENABLED:
        return False
    raw = os.environ.get("MEMORY_DEBUG_CONSOLE", "1").lower()
    return raw not in ("0", "false", "no", "off")


def _l1_file_probe(memory: Any, tenant_id: str, user_id: str) -> dict[str, Any]:
    hot = getattr(memory, "_hot", None)
    if hot is None:
        return {}
    out: dict[str, Any] = {}
    for label, fn_name, args in (
        ("memory_md", "_memory_path", (tenant_id,)),
        ("user_md", "_user_path", (tenant_id, user_id)),
        ("pending_jsonl", "_pending_path", (tenant_id, user_id)),
    ):
        fn = getattr(hot, fn_name, None)
        if not callable(fn):
            continue
        try:
            path = fn(*args)
            p = Path(path)
            out[label] = {
                "path": str(p),
                "exists": p.is_file(),
                "bytes": p.stat().st_size if p.is_file() else 0,
            }
        except OSError:
            out[label] = {"path": str(path), "exists": False, "bytes": 0}
    return out


def _hot_store_info(memory: Any) -> dict[str, Any]:
    hot = getattr(memory, "_hot", None)
    if hot is None:
        return {"backend": "unknown"}
    store = getattr(hot, "store_dir", None)
    backend = "relational"
    if store is not None and str(store).startswith("relational:"):
        pass
    elif store is not None:
        backend = "file"
    return {
        "backend": backend,
        "store_dir": str(store) if store is not None else None,
    }


def _archive_info(ctx: RunContext) -> dict[str, Any]:
    extra = ctx.extra or {}
    relational = extra.get("relational")
    summary = extra.get("memory_config_summary") or {}
    info: dict[str, Any] = {
        "archive_backend": summary.get("archive_backend")
        or load_memory_config().get("archive_backend", "sqlite"),
    }
    if relational is not None:
        info["adapter"] = type(relational).__name__
        if hasattr(relational, "_db_path"):
            info["sqlite_path"] = str(relational._db_path)
        elif hasattr(relational, "database"):
            info["pg_database"] = getattr(relational, "database", None)
    return info


def _turn_buffer_info(ctx: RunContext) -> dict[str, Any]:
    buf = ctx.turn_buffer
    if buf is None:
        return {"enabled": False}
    pending_fn = getattr(buf, "pending_count", None)
    flush_size = getattr(buf, "_flush_size", None) or getattr(buf, "flush_size", None)
    pending_count = pending_fn() if callable(pending_fn) else pending_fn
    return {
        "enabled": True,
        "flush_size": flush_size,
        "pending_count": pending_count,
        "adapter": type(buf).__name__,
    }


def _embedding_backend_label(ctx: RunContext, cfg: dict[str, Any]) -> str:
    models = getattr(ctx, "models", None)
    if models is not None:
        try:
            return models.get_embedding_version_key("embedding")
        except Exception:
            pass
    return str(cfg.get("embedding_role", "embedding"))


def _collect_l0_status(ctx: RunContext, cfg: dict[str, Any]) -> dict[str, Any]:
    extra = ctx.extra or {}
    state = extra.get("_l0_context_state")
    compressor = extra.get("context_compressor")
    last_savings = None
    if compressor is not None:
        last_savings = getattr(compressor, "last_savings_pct", None)
    if last_savings is None and isinstance(state, dict):
        last_savings = state.get("savings_pct")
    prompt_tokens = extra.get("_last_llm_prompt_tokens")
    l0_cfg = extra.get("l0_config") or {}
    return {
        "enabled": bool(l0_cfg.get("l0_context_compress_enabled", True)),
        "state_present": isinstance(state, dict),
        "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else None,
        "compress_count": int(extra.get("_l0_compress_count") or 0),
        "last_savings_pct": round(float(last_savings) * 100, 1)
        if last_savings is not None
        else None,
        "continuation": extra.get("_l0_compression_continuation"),
        "context_window_tokens": l0_cfg.get("l0_context_window_tokens")
        or cfg.get("l0_context_window_tokens"),
    }


def _collect_l1_hermes_status(memory: Any, req: Any, snap: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "frozen": bool(getattr(snap, "frozen", False)),
    }
    l1 = getattr(memory, "_l1", None)
    if l1 is None:
        return out
    try:
        live = l1.load_from_disk(req.tenant_id, req.user_id)
        mem_entries = live.get("memory") or []
        user_entries = live.get("user") or []
        mem_chars = l1._char_count(req.tenant_id, req.user_id, "memory")
        user_chars = l1._char_count(req.tenant_id, req.user_id, "user")
        mem_limit = l1.memory_char_limit
        user_limit = l1.user_char_limit
        out.update(
            {
                "memory_entry_count": len(mem_entries),
                "user_entry_count": len(user_entries),
                "memory_usage_pct": min(
                    100, int((mem_chars / mem_limit) * 100)
                )
                if mem_limit > 0
                else 0,
                "user_usage_pct": min(
                    100, int((user_chars / user_limit) * 100)
                )
                if user_limit > 0
                else 0,
                "memory_chars": mem_chars,
                "user_chars": user_chars,
            }
        )
    except Exception:
        pass
    key_fn = getattr(memory, "_session_snapshot_key", None)
    frozen_map = getattr(memory, "_frozen_snapshots", None)
    if callable(key_fn) and isinstance(frozen_map, dict):
        out["frozen_snapshot_cached"] = key_fn(req) in frozen_map
    return out


def _collect_l2_search_stats(memory: Any, req: Any) -> dict[str, Any]:
    getter = getattr(memory, "get_last_session_search_stats", None)
    if not callable(getter):
        return {}
    stats = getter(req)
    return dict(stats) if isinstance(stats, dict) else {}


def _collect_l4_provider_status(
    memory: Any, cfg: dict[str, Any], ctx: RunContext
) -> dict[str, Any]:
    ext = getattr(memory, "_external", None)
    provider = type(ext).__name__ if ext is not None else "none"
    inner = getattr(ext, "_inner", None)
    if inner is not None:
        provider = type(inner).__name__
    available = provider not in ("NoOpExternalMemoryAdapter", "none", "NoneType")
    cb_state = "n/a"
    cache = (ctx.extra or {}).get("cache")
    if cache is not None and hasattr(cache, "health"):
        try:
            health = cache.health()
            if isinstance(health, dict):
                cb_state = str(
                    health.get("circuit_breaker_state") or cb_state
                )
        except Exception:
            pass
    if ctx.models is not None:
        try:
            info = ctx.models.get_model_info("main_llm")
            if getattr(info, "circuit_open", False):
                cb_state = "open"
            elif cb_state == "n/a":
                cb_state = "closed"
        except Exception:
            pass
    return {
        "provider": provider,
        "available": available,
        "circuit_breaker_state": cb_state,
        "backend": cfg.get("external_profiles_backend", "file"),
    }


def _checkpointer_info(ctx: RunContext) -> dict[str, Any]:
    cp = ctx.checkpointer
    out: dict[str, Any] = {"relational": None, "langgraph": None}
    if cp is not None:
        out["relational"] = {
            "adapter": type(cp).__name__,
            "table": getattr(cp, "TABLE", None),
        }
    try:
        from app.runtime.adapters.langgraph import checkpointer as lg_cp

        if getattr(lg_cp, "_PG_SAVER", None) is not None:
            out["langgraph"] = {
                "backend": "postgresql",
                "adapter": type(lg_cp._PG_SAVER).__name__,
            }
        elif getattr(lg_cp, "_CHECKPOINTER", None) is not None:
            out["langgraph"] = {
                "backend": "memory_or_sqlite",
                "adapter": type(lg_cp._CHECKPOINTER).__name__,
            }
    except Exception:
        pass
    extra = ctx.extra or {}
    if extra.get("langgraph_checkpoint_path"):
        out["langgraph_sqlite_path"] = extra.get("langgraph_checkpoint_path")
    return out


async def collect_memory_runtime_status(
    ctx: RunContext,
    *,
    event: str = "snapshot",
    session_turns_limit: int = 5,
    verbose: bool | None = None,
) -> dict[str, Any]:
    """采集 L1/L2/L3/L4 + 基础设施完整运行状态（verbose 时含文本预览）。"""
    detailed = _VERBOSE if verbose is None else verbose
    req = ctx.request
    memory = ctx.memory
    extra = ctx.extra or {}
    cfg = load_memory_config()

    status: dict[str, Any] = {
        "event": event,
        "timestamp_ms": int(time.time() * 1000),
        "request": {
            "tenant_id": req.tenant_id,
            "user_id": req.user_id,
            "session_id": req.session_id,
            "trace_id": getattr(req, "trace_id", None),
            "channel": getattr(req, "channel", None),
        },
        "config": {
            **memory_config_summary(),
            "config_path": cfg.get("_config_path"),
            "memory_config_env": os.environ.get("MEMORY_CONFIG"),
            "chat_profile_extra": extra.get("memory_config_summary"),
        },
        "layers": {},
        "infrastructure": {},
        "metrics": {},
    }

    if memory is None:
        status["layers"]["error"] = "MemoryPort 未注入"
        return status

    status["layers"]["L0"] = _collect_l0_status(ctx, cfg)

    # L1
    try:
        snap = memory.compose_prompt_snapshot(req)
        pending = list_pending_l1_deltas(ctx)
        l1: dict[str, Any] = {
            "hash": snap.hash,
            "chars": len(snap.memory_text or ""),
            "pending_count": len(pending),
            "pending_keys": [p.get("key") for p in pending[:10]],
            **_hot_store_info(memory),
            **_collect_l1_hermes_status(memory, req, snap),
        }
        if detailed:
            l1["memory_preview"] = _preview(snap.memory_text, 800)
            l1["files"] = _l1_file_probe(memory, req.tenant_id, req.user_id)
            l1["pending_deltas"] = [
                {
                    "key": p.get("key"),
                    "value_preview": _preview(str(p.get("value") or ""), 120),
                    "source": p.get("source"),
                }
                for p in pending[:20]
            ]
        status["layers"]["L1"] = l1
    except Exception as exc:
        status["layers"]["L1"] = {"error": type(exc).__name__, "msg": str(exc)[:200]}

    # L2
    l2: dict[str, Any] = {**_archive_info(ctx)}
    search_stats = _collect_l2_search_stats(memory, req)
    if search_stats:
        l2["last_search"] = search_stats
    turn_limit = 200 if detailed else session_turns_limit * 2
    try:
        rows = await memory.list_turns(req, limit=turn_limit)
        l2["session_turn_rows"] = len(rows)
        l2["recent_roles"] = [r.get("role") for r in rows[-session_turns_limit:]]
        buf = ctx.turn_buffer
        if buf is not None and hasattr(buf, "pending_turns_for"):
            pending_turns = buf.pending_turns_for(req)
            l2["turn_buffer_pending"] = len(pending_turns)
            if detailed and pending_turns:
                l2["turn_buffer_preview"] = [
                    {
                        "role": t.get("role"),
                        "chars": len(t.get("content") or ""),
                        "preview": _preview(t.get("content"), 100),
                    }
                    for t in pending_turns[-6:]
                ]
        if detailed:
            l2["recent_turns"] = [
                {
                    "role": r.get("role"),
                    "ts": r.get("ts"),
                    "chars": len(r.get("content") or ""),
                    "preview": _preview(r.get("content"), 150),
                }
                for r in rows[-12:]
            ]
        list_sessions = getattr(memory, "list_sessions", None)
        if detailed and list_sessions is not None:
            try:
                sessions = await list_sessions(req, limit=8)
                l2["user_sessions"] = [
                    {
                        "session_id": s.get("session_id"),
                        "status": s.get("status"),
                        "started_at": s.get("started_at"),
                    }
                    for s in sessions[:8]
                ]
            except Exception:
                pass
    except Exception as exc:
        l2["error"] = type(exc).__name__
    status["layers"]["L2"] = l2

    # L3
    l3: dict[str, Any] = {}
    try:
        lister = getattr(memory, "list_skills", None)
        if lister is not None:
            skills = await lister(req)
            l3["published_count"] = len(skills)
            l3["skill_ids"] = [s.get("skill_id") for s in skills[:8]]
            if detailed:
                l3["skills"] = [
                    {
                        "skill_id": s.get("skill_id"),
                        "title": s.get("title"),
                        "status": s.get("status"),
                        "usage_count": s.get("usage_count"),
                    }
                    for s in skills[:12]
                ]
    except Exception as exc:
        l3["error"] = type(exc).__name__
    status["layers"]["L3"] = l3

    # L4
    l4: dict[str, Any] = {
        "backend": cfg.get("external_profiles_backend", "file"),
        "cache_ttl": cfg.get("external_profile_cache_ttl", 0),
        "merge_on_finalize": bool(cfg.get("external_merge_on_finalize")),
        "profiles_dir": cfg.get("external_profiles_dir"),
        **_collect_l4_provider_status(memory, cfg, ctx),
    }
    try:
        fetcher = getattr(memory, "fetch_profile_facts", None)
        if fetcher is not None:
            facts = await fetcher(req.tenant_id, req.user_id)
            l4["facts_count"] = len(facts)
            l4["fact_keys"] = [_fact_row(f).get("key") for f in facts[:8]]
            if detailed:
                l4["facts"] = [_fact_row(f) for f in facts[:12]]
    except Exception as exc:
        l4["error"] = type(exc).__name__
    status["layers"]["L4"] = l4

    status["infrastructure"] = {
        "turn_buffer": _turn_buffer_info(ctx),
        "checkpointer": _checkpointer_info(ctx),
        "cold_archive": bool(cfg.get("enable_cold_archive")),
        "session_vector_index": bool(cfg.get("enable_session_vector_index")),
        "embedding_backend": _embedding_backend_label(ctx, cfg),
        "object_store": type(extra.get("object_store")).__name__
        if extra.get("object_store")
        else None,
        "cache": type(extra.get("cache")).__name__ if extra.get("cache") else None,
    }

    status["rag"] = {
        "port_present": ctx.rag is not None,
        "chroma_dir": extra.get("rag_chroma_dir"),
        "tenant_id": extra.get("rag_tenant_id"),
    }

    try:
        status["metrics"] = get_memory_metric_stats(ctx)
    except Exception:
        status["metrics"] = {}

    if detailed:
        try:
            from app.agents.orchestration.chat_config import load_chat_config

            status["chat_config"] = chat_config_verify_snapshot(load_chat_config())
        except Exception:
            pass
        status["verbose"] = True

    return status


async def log_turn_trace(
    ctx: RunContext,
    *,
    phase: str,
    user_message: str = "",
    extra: Optional[dict[str, Any]] = None,
    force: bool = False,
) -> dict[str, Any]:
    """单轮对话阶段追踪（turn_start / turn_done）。"""
    payload = {
        "phase": phase,
        "user_message_preview": _preview(user_message, 200),
        "user_message_chars": len(user_message or ""),
    }
    if extra:
        payload.update(extra)
    if phase.endswith("_done") or phase == "turn_done":
        data = await collect_memory_runtime_status(ctx, event=phase)
        data["turn"] = payload
    else:
        data = {
            "event": phase,
            "timestamp_ms": int(time.time() * 1000),
            "request": {
                "tenant_id": ctx.request.tenant_id,
                "user_id": ctx.request.user_id,
                "session_id": ctx.request.session_id,
            },
            "turn": payload,
        }
    trace_write(
        hypothesis_id="MEM-TURN",
        location="memory_runtime_debug.log_turn_trace",
        message=f"turn trace: {phase}",
        data=data,
        run_id=ctx.request.session_id,
        force=force,
    )
    return data


async def log_memory_runtime_status(
    ctx: RunContext,
    *,
    event: str,
    hypothesis_id: str = "MEM-RUNTIME",
    location: str = "memory_runtime_debug",
    run_id: str = "default",
    console: bool = False,
    extra: Optional[dict[str, Any]] = None,
    force: bool = False,
) -> dict[str, Any]:
    """采集并写入 NDJSON；可选控制台摘要。force=True 时即使未开 debug 也写入。"""
    data = await collect_memory_runtime_status(ctx, event=event)
    if extra:
        data["extra"] = extra
    _write_ndjson(
        hypothesis_id=hypothesis_id,
        location=location,
        message=f"memory runtime: {event}",
        data=data,
        run_id=run_id,
        force=force,
    )
    if console and (_ENABLED or force):
        layers = data.get("layers") or {}
        l1 = layers.get("L1") or {}
        l2 = layers.get("L2") or {}
        print(
            f"\n[memory-runtime:{event}] "
            f"L1 hash={l1.get('hash')} chars={l1.get('chars')} pending={l1.get('pending_count')} | "
            f"L2 rows={l2.get('session_turn_rows')} archive={l2.get('archive_backend')} | "
            f"log→ {debug_log_path()}\n"
        )
    return data


def format_memory_runtime_summary(data: dict[str, Any]) -> str:
    """人类可读摘要（REPL /status 用）。"""
    layers = data.get("layers") or {}
    infra = data.get("infrastructure") or {}
    l1 = layers.get("L1") or {}
    l2 = layers.get("L2") or {}
    l3 = layers.get("L3") or {}
    l4 = layers.get("L4") or {}
    l0 = layers.get("L0") or {}
    cfg = data.get("config") or {}
    cp = infra.get("checkpointer") or {}
    lines = [
        "--- memory runtime ---",
        f"  event={data.get('event')}  tenant={data.get('request', {}).get('tenant_id')}",
        f"  L0: enabled={l0.get('enabled')} tokens={l0.get('prompt_tokens')} "
        f"compress={l0.get('compress_count')} savings={l0.get('last_savings_pct')}% "
        f"state={l0.get('state_present')}",
        f"  L1: backend={l1.get('backend')} hash={l1.get('hash')} chars={l1.get('chars')} "
        f"pending={l1.get('pending_count')} usage={l1.get('memory_usage_pct')}% "
        f"entries={l1.get('memory_entry_count')}",
        f"  L2: archive={l2.get('archive_backend')} turns={l2.get('session_turn_rows')} "
        f"buf_pending={l2.get('turn_buffer_pending')}",
        f"  L3: skills={l3.get('published_count', '?')}",
        f"  L4: provider={l4.get('provider')} avail={l4.get('available')} "
        f"cb={l4.get('circuit_breaker_state')} facts={l4.get('facts_count', '?')}",
        f"  infra: cold={infra.get('cold_archive')} vector={infra.get('session_vector_index')} embed={infra.get('embedding_backend')}",
        f"  checkpointer: rel={bool(cp.get('relational'))} lg={cp.get('langgraph') or cp.get('langgraph_sqlite_path')}",
        f"  config: {cfg.get('config_path') or cfg.get('memory_config')}",
        f"  debug log: {debug_log_path()}",
    ]
    metrics = data.get("metrics") or {}
    if metrics:
        lines.append("  metrics:")
        for name, stats in metrics.items():
            lines.append(f"    {name}: count={stats.get('count')} sum={stats.get('sum')}")
    lines.append("")
    return "\n".join(lines)


def format_memory_runtime_detailed(data: dict[str, Any]) -> str:
    """完整人类可读调试报告（REPL /debug-memory、轮次摘要）。"""
    req = data.get("request") or {}
    layers = data.get("layers") or {}
    infra = data.get("infrastructure") or {}
    rag = data.get("rag") or {}
    cc = data.get("chat_config") or {}
    turn = data.get("turn") or data.get("turn_result") or {}
    l1 = layers.get("L1") or {}
    l2 = layers.get("L2") or {}
    l3 = layers.get("L3") or {}
    l4 = layers.get("L4") or {}
    l0 = layers.get("L0") or {}
    cp = infra.get("checkpointer") or {}
    tb = infra.get("turn_buffer") or {}

    lines = [
        "========== 记忆子系统调试详情 ==========",
        f"event={data.get('event')}  session={req.get('session_id')}  "
        f"tenant={req.get('tenant_id')}  user={req.get('user_id')}",
        "",
        "── L0 上下文压缩 ──",
        f"  enabled={l0.get('enabled')}  window={l0.get('context_window_tokens')}",
        f"  prompt_tokens={l0.get('prompt_tokens')}  compress_count={l0.get('compress_count')}",
        f"  last_savings={l0.get('last_savings_pct')}%  state_present={l0.get('state_present')}",
        f"  continuation={l0.get('continuation') or '—'}",
        "",
        "── L1 热记忆 ──",
        f"  backend={l1.get('backend')}  store={l1.get('store_dir')}",
        f"  hash={l1.get('hash')}  chars={l1.get('chars')}  pending={l1.get('pending_count')}",
        f"  frozen={l1.get('frozen')}  cached={l1.get('frozen_snapshot_cached')}",
        f"  memory: entries={l1.get('memory_entry_count')} usage={l1.get('memory_usage_pct')}%",
        f"  user: entries={l1.get('user_entry_count')} usage={l1.get('user_usage_pct')}%",
    ]
    if l1.get("memory_preview"):
        lines.append(f"  preview:\n    {l1.get('memory_preview')}")
    files = l1.get("files") or {}
    for label, info in files.items():
        if isinstance(info, dict):
            lines.append(
                f"  file[{label}]: exists={info.get('exists')} "
                f"bytes={info.get('bytes')} path={info.get('path')}"
            )
    pending = l1.get("pending_deltas") or []
    if pending:
        lines.append("  pending_deltas:")
        for p in pending[:10]:
            lines.append(f"    - {p.get('key')}={p.get('value_preview')} ({p.get('source')})")

    lines.extend([
        "",
        "── L2 会话归档 ──",
        f"  backend={l2.get('archive_backend')}  db={l2.get('sqlite_path') or l2.get('pg_database')}",
        f"  turns_loaded={l2.get('session_turn_rows')}  buffer_pending={l2.get('turn_buffer_pending')}",
        f"  turn_buffer: enabled={tb.get('enabled')} flush_size={tb.get('flush_size')}",
    ])
    last_search = l2.get("last_search") or {}
    if last_search:
        lines.append(
            f"  last_search: mode={last_search.get('mode')} hit={last_search.get('hit')} "
            f"cached={last_search.get('cached')} chars={last_search.get('chars')} "
            f"q={last_search.get('query_preview')!r}"
        )
    for t in (l2.get("recent_turns") or [])[-6:]:
        lines.append(
            f"    [{t.get('role')}] {t.get('chars')}ch  "
            f"{(t.get('preview') or '')[:100]}"
        )

    lines.extend([
        "",
        "── L3 技能 ──",
        f"  published={l3.get('published_count', '?')}",
    ])
    for s in (l3.get("skills") or [])[:8]:
        lines.append(
            f"    - {s.get('skill_id')}: {s.get('title')} "
            f"({s.get('status')}, used={s.get('usage_count')})"
        )

    lines.extend([
        "",
        "── L4 外部画像 ──",
        f"  provider={l4.get('provider')}  available={l4.get('available')}  "
        f"cb={l4.get('circuit_breaker_state')}",
        f"  backend={l4.get('backend')}  dir={l4.get('profiles_dir')}  "
        f"merge_on_finalize={l4.get('merge_on_finalize')}",
        f"  facts={l4.get('facts_count', '?')}",
    ])
    for f in (l4.get("facts") or [])[:8]:
        lines.append(f"    - {f.get('key')}: {f.get('value')}  [{f.get('source')}]")

    lines.extend([
        "",
        "── RAG ──",
        f"  port={rag.get('port_present')}  tenant={rag.get('tenant_id')}  chroma={rag.get('chroma_dir')}",
        "",
        "── 基础设施 ──",
        f"  cold_archive={infra.get('cold_archive')}  vector_index={infra.get('session_vector_index')}",
        f"  checkpointer_rel={bool(cp.get('relational'))}  lg={cp.get('langgraph') or cp.get('langgraph_sqlite_path')}",
    ])
    if cc:
        lines.extend([
            "",
            "── Chat 配置 ──",
            f"  rag_gating={cc.get('enable_rag_gating')} min_chars={cc.get('rag_gating_min_chars')}",
            f"  max_history_turns={cc.get('max_history_turns')} max_history_chars={cc.get('max_history_chars')}",
            f"  memory_tools={cc.get('enable_memory_tools')} hitl={cc.get('remember_require_hitl')}",
            f"  l1_extract_keys={cc.get('l1_extract_allowed_keys')}",
        ])
    if turn:
        lines.extend([
            "",
            "── 本轮对话 ──",
            f"  user: {(turn.get('user_message_preview') or turn.get('user_message') or '')[:120]}",
            f"  rag: evidence={turn.get('evidence_count')} empty={turn.get('rag_empty')} "
            f"history_turns={turn.get('history_turns')}",
            f"  assistant: {turn.get('assistant_chars')} chars",
        ])
        if turn.get("assistant_preview"):
            lines.append(f"  assistant_preview: {turn.get('assistant_preview')[:200]}…")
    timeline = data.get("layer_trigger_timeline") or []
    if timeline:
        lines.append("")
        lines.append("── 本轮触发链 ──")
        for t in timeline:
            flag = "ON" if t.get("triggered") else "SKIP"
            lines.append(
                f"  [{flag}] {t.get('layer')}::{t.get('action')}  {t.get('reason') or ''}"
            )

    metrics = data.get("metrics") or {}
    if metrics:
        lines.append("")
        lines.append("── 指标 ──")
        for name, stats in metrics.items():
            lines.append(f"  {name}: count={stats.get('count')} sum={stats.get('sum')}")

    lines.extend([
        "",
        f"NDJSON 日志: {debug_log_path()}",
        "========================================",
        "",
    ])
    return "\n".join(lines)


def format_post_turn_debug(
    status: dict[str, Any],
    *,
    user_message: str = "",
    evidence_count: int = 0,
    rag_empty: bool = True,
    history_turns: int = 0,
    assistant_chars: int = 0,
    layer_trigger_timeline: Optional[list] = None,
) -> str:
    """轮次结束后的控制台详细摘要。"""
    status = dict(status)
    status["turn_result"] = {
        "user_message_preview": _preview(user_message, 120),
        "evidence_count": evidence_count,
        "rag_empty": rag_empty,
        "history_turns": history_turns,
        "assistant_chars": assistant_chars,
    }
    if layer_trigger_timeline:
        status["layer_trigger_timeline"] = layer_trigger_timeline
    return format_memory_runtime_detailed(status)
