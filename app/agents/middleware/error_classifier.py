# -*- coding: utf-8 -*-
"""ErrorClassifierMiddleware：节点异常分类与指标。"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from core.composition.run_context import RunContext

from app.agents.observability.graph_metrics import record_graph_node_error


def classify_exception(error: Exception) -> str:
    """将异常映射为企业可观测 error_type。"""
    if isinstance(error, PermissionError):
        return "policy_denied"
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return "llm_timeout"

    msg = str(error).lower()
    type_name = type(error).__name__.lower()

    if "timeout" in msg or "timed out" in msg:
        return "llm_timeout"
    if "memory" in msg or "session_search" in msg or "archive" in msg:
        return "memory_error"
    if "tool" in type_name or "tool" in msg:
        return "tool_error"
    if isinstance(error, MemoryError):
        return "memory_error"
    return "unknown"


class ErrorClassifierMiddleware:
    """节点 on_exit 记录 graph.node.errors_total（带 error_type）。"""

    @property
    def name(self) -> str:
        return "error_classifier"

    async def on_enter(
        self,
        node_name: str,
        state: Any,
        config: Any,
    ) -> Dict[str, Any]:
        return {}

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
        if error is None:
            return

        error_type = classify_exception(error)
        configurable = (config or {}).get("configurable") or {}
        run_ctx: RunContext | None = configurable.get("run_ctx")
        tenant_id = ""
        if run_ctx is not None:
            req = getattr(run_ctx, "request", None)
            if req is not None:
                tenant_id = str(getattr(req, "tenant_id", "") or "")

        record_graph_node_error(
            run_ctx,
            node=node_name,
            error_type=error_type,
            tenant_id=tenant_id,
        )
