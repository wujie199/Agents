# -*- coding: utf-8 -*-
"""TracingMiddleware：trace_id + span 注入，跨节点链路追踪。"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional


class TracingMiddleware:
    """注入 trace_id 和 span 信息，记录节点耗时。"""

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
        span_id = f"{node_name}:{uuid.uuid4().hex[:8]}"
        return {
            "trace_id": trace_id,
            "span_id": span_id,
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
