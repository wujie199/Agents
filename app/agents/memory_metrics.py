# -*- coding: utf-8 -*-
"""记忆子系统指标（ObservabilityPort + 可选 Prometheus）。"""

from __future__ import annotations

from typing import Any, Optional

from core.composition.run_context import RunContext


def _obs(ctx: Optional[RunContext]) -> Any:
    if ctx is None:
        return None
    return ctx.observability


def _metric_stats_from_obs(obs: Any) -> dict:
    if obs is None or not hasattr(obs, "get_metrics"):
        return {}
    names = [
        "memory.chat.turn",
        "memory.chat.evidence_count",
        "memory.l1.confirm",
        "memory.purge",
        "memory.session_search",
        "cache.rag.hit",
        "cache.rag.miss",
        "cache.redis.hit",
        "cache.redis.miss",
        "cache.redis.fallback",
    ]
    out: dict = {}
    raw = obs.get_metrics()
    for name in names:
        if name in raw and hasattr(obs, "get_metric_stats"):
            stats = obs.get_metric_stats(name)
            if stats:
                out[name] = stats
    return out


def record_memory_metric(
    ctx: Optional[RunContext],
    name: str,
    value: float = 1.0,
    *,
    tags: Optional[dict] = None,
) -> None:
    obs = _obs(ctx)
    if obs is None or not hasattr(obs, "record_metric"):
        return
    merged = {"subsystem": "memory"}
    if tags:
        merged.update(tags)
    obs.record_metric(name, value, merged)


def record_chat_turn(
    ctx: RunContext,
    *,
    evidence_count: int,
    rag_empty: bool,
    history_turns: int,
    engine: str = "unknown",
) -> None:
    record_memory_metric(ctx, "memory.chat.turn", 1.0, tags={"engine": engine})
    record_memory_metric(
        ctx,
        "memory.chat.evidence_count",
        float(evidence_count),
        tags={"rag_empty": str(rag_empty)},
    )
    record_memory_metric(
        ctx, "memory.chat.history_turns", float(history_turns)
    )


def record_l1_confirm(ctx: RunContext, count: int) -> None:
    record_memory_metric(ctx, "memory.l1.confirm", float(count))


def record_l1_pending(ctx: RunContext, count: int) -> None:
    record_memory_metric(ctx, "memory.l1.pending", float(count))


def record_purge(ctx: RunContext, *, scope: str, count: float = 1.0) -> None:
    record_memory_metric(
        ctx, "memory.purge", count, tags={"scope": scope}
    )


def record_session_search(ctx: RunContext, *, hit: bool) -> None:
    record_memory_metric(
        ctx,
        "memory.session_search",
        1.0,
        tags={"hit": str(hit)},
    )
    record_cache_metric(ctx, subsystem="session_search", hit=hit)


def record_cache_metric(
    ctx: Optional[RunContext],
    *,
    subsystem: str,
    hit: bool,
) -> None:
    name = f"cache.{subsystem}.{'hit' if hit else 'miss'}"
    record_memory_metric(ctx, name, 1.0, tags={"subsystem": subsystem})


def _record_stats_delta(
    ctx: RunContext,
    key: str,
    stats: dict,
    *,
    hit_field: str,
    miss_field: Optional[str] = None,
    subsystem: str,
) -> None:
    extra = dict(getattr(ctx, "extra", None) or {})
    last = extra.get(key) or {}
    hit_delta = int(stats.get(hit_field, 0)) - int(last.get(hit_field, 0))
    if miss_field:
        miss_delta = int(stats.get(miss_field, 0)) - int(last.get(miss_field, 0))
    else:
        miss_delta = 0
    for _ in range(max(0, hit_delta)):
        record_cache_metric(ctx, subsystem=subsystem, hit=True)
    for _ in range(max(0, miss_delta)):
        record_cache_metric(ctx, subsystem=subsystem, hit=False)
    extra[key] = dict(stats)
    if hasattr(ctx, "extra"):
        ctx.extra.update(extra)


def record_rag_cache_stats(ctx: RunContext) -> None:
    rag = ctx.rag
    if rag is None or not hasattr(rag, "get_cache_stats"):
        return
    _record_stats_delta(
        ctx,
        "_rag_cache_stats_last",
        rag.get_cache_stats(),
        hit_field="hit",
        miss_field="miss",
        subsystem="rag",
    )


def record_redis_cache_stats(ctx: RunContext) -> None:
    cache = (getattr(ctx, "extra", None) or {}).get("cache")
    if cache is None or not hasattr(cache, "get_stats"):
        return
    stats = cache.get_stats()
    extra = getattr(ctx, "extra", None) or {}
    last = extra.get("_redis_cache_stats_last") or {}
    fb_delta = int(stats.get("fallback_used", 0)) - int(last.get("fallback_used", 0))
    _record_stats_delta(
        ctx,
        "_redis_cache_stats_last",
        stats,
        hit_field="get_hit",
        miss_field="get_miss",
        subsystem="redis",
    )
    for _ in range(max(0, fb_delta)):
        record_memory_metric(ctx, "cache.redis.fallback", 1.0)


def get_memory_metric_stats(ctx: RunContext) -> dict:
    return _metric_stats_from_obs(_obs(ctx))


def get_memory_metric_stats_from_obs(obs: Any) -> dict:
    """从共享 ObservabilityPort 读取 memory.* 指标。"""
    return _metric_stats_from_obs(obs)


def prometheus_text(
    ctx: Optional[RunContext] = None,
    *,
    observability: Any = None,
    cache_stats: Optional[dict] = None,
) -> str:
    """简易 Prometheus 文本格式（memory.* + cache.*）。"""
    obs = observability if observability is not None else _obs(ctx)
    if obs is None or not hasattr(obs, "get_metrics"):
        lines: list[str] = ["# no metrics\n"]
    else:
        lines = []
        for name, values in obs.get_metrics().items():
            if not (
                name.startswith("memory.")
                or name.startswith("cache.")
            ):
                continue
            total = sum(values)
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name}_total {total}")
            lines.append(f"{name}_count {len(values)}")
    if cache_stats:
        for key, value in cache_stats.items():
            if key == "circuit_breaker_state":
                continue
            metric = f"cache_redis_{key}"
            lines.append(f"# TYPE {metric} gauge")
            lines.append(f"{metric} {value}")
    return "\n".join(lines) + "\n"
