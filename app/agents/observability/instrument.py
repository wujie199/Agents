# -*- coding: utf-8 -*-
"""Span 上下文管理器与指标辅助。"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

from core.composition.run_context import RunContext
from core.ports.observability import Layer


def _obs(ctx: Optional[RunContext]) -> Any:
    if ctx is None:
        return None
    return ctx.observability


def _trace_id(ctx: Optional[RunContext]) -> str:
    if ctx is None:
        return ""
    req = getattr(ctx, "request", None)
    if req is None:
        return ""
    return str(getattr(req, "trace_id", "") or "")


def _identity_attrs(ctx: Optional[RunContext]) -> Dict[str, Any]:
    if ctx is None:
        return {}
    req = getattr(ctx, "request", None)
    if req is None:
        return {}
    out: Dict[str, Any] = {}
    for key in ("tenant_id", "user_id", "session_id"):
        val = getattr(req, key, None)
        if val:
            out[key] = str(val)
    return out


@asynccontextmanager
async def span_ctx(
    ctx: Optional[RunContext],
    name: str,
    layer: Layer,
    attributes: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """创建 observability span；无 obs 时 no-op。"""
    attrs: Dict[str, Any] = dict(attributes or {})
    attrs.update(_identity_attrs(ctx))
    obs = _obs(ctx)
    span = None
    if obs is not None and hasattr(obs, "start_span"):
        span = obs.start_span(
            _trace_id(ctx),
            name,
            layer,
            attributes=attrs,
        )
    t0 = time.perf_counter()
    try:
        yield attrs
    finally:
        attrs.setdefault("duration_ms", round((time.perf_counter() - t0) * 1000, 2))
        if span is not None and obs is not None and hasattr(obs, "end_span"):
            obs.end_span(span, attributes=attrs)


def record_span_metric(
    ctx: Optional[RunContext],
    name: str,
    value: float,
    tags: Optional[Dict[str, Any]] = None,
) -> None:
    obs = _obs(ctx)
    if obs is None or not hasattr(obs, "record_metric"):
        return
    obs.record_metric(name, value, tags or {})
