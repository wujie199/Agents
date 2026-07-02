# -*- coding: utf-8 -*-
"""LangGraph / Agent 子步骤指标。"""

from __future__ import annotations

from typing import Any, Optional

from core.composition.run_context import RunContext

from app.agents.observability.instrument import record_span_metric


def record_graph_node_duration(
    ctx: Optional[RunContext],
    duration_ms: float,
    *,
    node: str,
    tenant_id: str = "",
    error: bool = False,
    slow: bool = False,
) -> None:
    record_span_metric(
        ctx,
        "graph.node.duration_ms",
        duration_ms,
        tags={
            "node": node,
            "tenant_id": tenant_id,
            "error": str(error),
            "slow": str(slow),
        },
    )


def record_tool_call(
    ctx: Optional[RunContext],
    duration_ms: float,
    *,
    tool_name: str,
    success: bool,
) -> None:
    record_span_metric(
        ctx,
        "agent.tool.calls_total",
        1.0,
        tags={"tool_name": tool_name, "success": str(success)},
    )
    record_span_metric(
        ctx,
        "agent.tool.duration_ms",
        duration_ms,
        tags={"tool_name": tool_name, "success": str(success)},
    )


def record_llm_call(
    ctx: Optional[RunContext],
    duration_ms: float,
    *,
    prompt_tokens: Optional[int] = None,
) -> None:
    record_span_metric(ctx, "agent.llm.duration_ms", duration_ms)
    if prompt_tokens is not None:
        record_span_metric(
            ctx,
            "agent.llm.prompt_tokens",
            float(prompt_tokens),
        )


def record_l0_compress_triggered(ctx: Optional[RunContext]) -> None:
    record_span_metric(ctx, "memory.l0.compress_triggered", 1.0)


def record_graph_node_error(
    ctx: Optional[RunContext],
    *,
    node: str,
    error_type: str,
    tenant_id: str = "",
) -> None:
    record_span_metric(
        ctx,
        "graph.node.errors_total",
        1.0,
        tags={
            "node": node,
            "error_type": error_type,
            "tenant_id": tenant_id,
        },
    )


_GRAPH_METRIC_NAMES = (
    "graph.node.duration_ms",
    "graph.node.errors_total",
    "agent.tool.calls_total",
    "agent.tool.duration_ms",
    "agent.llm.duration_ms",
    "agent.llm.prompt_tokens",
    "memory.l0.compress_triggered",
)


def graph_metric_names() -> tuple[str, ...]:
    return _GRAPH_METRIC_NAMES
