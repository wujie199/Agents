# -*- coding: utf-8 -*-
"""MetricsMiddleware：图节点耗时与错误指标。"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from core.composition.run_context import RunContext

from app.agents.observability.graph_metrics import record_graph_node_duration


class MetricsMiddleware:
    """记录 graph.node.duration_ms 指标。"""

    def __init__(
        self,
        *,
        slow_threshold_ms: float = 3000,
        node_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self._slow_threshold_ms = slow_threshold_ms
        self._node_thresholds = node_thresholds or {}

    @property
    def name(self) -> str:
        return "metrics"

    async def on_enter(
        self,
        node_name: str,
        state: Any,
        config: Any,
    ) -> Dict[str, Any]:
        return {"_metrics_start": time.perf_counter()}

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
        ctx_extra = extra or {}
        t0 = ctx_extra.get("_metrics_start") or ctx_extra.get("_timing_start")
        if t0 is None:
            t0 = ctx_extra.get("_span_start")
        if t0 is None:
            return

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)
        threshold = self._node_thresholds.get(node_name, self._slow_threshold_ms)
        is_slow = duration_ms >= threshold

        configurable = (config or {}).get("configurable") or {}
        run_ctx: RunContext | None = configurable.get("run_ctx")
        tenant_id = ""
        if run_ctx is not None:
            req = getattr(run_ctx, "request", None)
            if req is not None:
                tenant_id = str(getattr(req, "tenant_id", "") or "")

        record_graph_node_duration(
            run_ctx,
            duration_ms,
            node=node_name,
            tenant_id=tenant_id,
            error=error is not None,
            slow=is_slow,
        )
