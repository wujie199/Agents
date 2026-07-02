# -*- coding: utf-8 -*-
"""TracingMiddleware：trace_id + span 注入，跨节点链路追踪。"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from core.composition.run_context import RunContext
from core.ports.observability import Layer


class TracingMiddleware:
    """注入 trace_id 和 span 信息，对接 ObservabilityPort。"""

    def __init__(self, *, trace_id_header: str = "trace_id") -> None:
        self._trace_id_header = trace_id_header

    @property
    def name(self) -> str:
        return "tracing"

    async def on_enter(
        self,
        node_name: str,
        state: Any,
        config: Any,
    ) -> Dict[str, Any]:
        configurable = (config or {}).get("configurable") or {}
        trace_id = configurable.get("trace_id") or str(uuid.uuid4())[:16]
        span_attributes = dict(configurable.get("_span_attributes") or {})
        span_attributes.setdefault("node", node_name)

        run_ctx: RunContext | None = configurable.get("run_ctx")
        obs = getattr(run_ctx, "observability", None) if run_ctx else None
        parent_span_id = configurable.get("_active_span_id")

        span = None
        span_id = f"{node_name}:{uuid.uuid4().hex[:8]}"
        if obs is not None and hasattr(obs, "start_span"):
            span = obs.start_span(
                trace_id=trace_id,
                name=f"graph.{node_name}",
                layer=Layer.WORKFLOW,
                parent_span_id=parent_span_id,
                attributes=span_attributes,
            )
            span_id = span.span_id
            if isinstance(config, dict):
                config["configurable"]["_active_span_id"] = span_id

        return {
            "trace_id": trace_id,
            "span_id": span_id,
            "_obs_span": span,
            "_span_start": time.perf_counter(),
        }

    async def on_exit(
        self,
        node_name: str,
        state: Any,
        config: Any,
        result: Any,
        *,
        error: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = extra or {}
        start = ctx.get("_span_start")
        if start is not None:
            duration_ms = (time.perf_counter() - start) * 1000
            ctx["duration_ms"] = round(duration_ms, 2)

        span = ctx.get("_obs_span")
        configurable = (config or {}).get("configurable") or {}
        run_ctx: RunContext | None = configurable.get("run_ctx")
        obs = getattr(run_ctx, "observability", None) if run_ctx else None

        exit_attrs: Dict[str, Any] = {"duration_ms": ctx.get("duration_ms", 0)}
        if error is not None:
            exit_attrs["error"] = str(error)

        if span is not None and obs is not None and hasattr(obs, "end_span"):
            obs.end_span(span, attributes=exit_attrs)
