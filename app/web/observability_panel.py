# -*- coding: utf-8 -*-
"""Web 侧栏：图节点耗时、工具调用、审计摘要。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.composition.run_context import RunContext

from app.agents.orchestration.chat_config import ObservabilityConfig, load_observability_config


def begin_turn_observability(ctx: RunContext) -> None:
    """新一轮对话开始前重置本轮观测快照。"""
    extra = getattr(ctx, "extra", None)
    if not isinstance(extra, dict):
        return
    extra["turn_audit"] = []
    extra["_web_agent_events_offset"] = len(extra.get("agent_events") or [])


def _fmt_ms(ms: Any) -> str:
    if ms is None:
        return "N/A"
    try:
        v = float(ms)
    except (TypeError, ValueError):
        return "N/A"
    if v < 1000:
        return f"{v:.0f}ms"
    return f"{v / 1000:.2f}s"


def _load_audit_from_file(
    session_id: str,
    audit_dir: str,
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = Path(audit_dir) / f"audit_{day}.jsonl"
    if not path.is_file():
        return []
    matched: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("session_id") or "") == session_id:
                matched.append(row)
    except OSError:
        return []
    return matched[-limit:]


def collect_turn_observability(
    ctx: RunContext,
    session_id: str,
    *,
    obs_cfg: Optional[ObservabilityConfig] = None,
) -> dict[str, Any]:
    """从 run_ctx.extra 与审计文件收集本轮监控快照。"""
    cfg = obs_cfg or load_observability_config()
    extra = getattr(ctx, "extra", None) or {}

    offset = int(extra.get("_web_agent_events_offset") or 0)
    all_events = list(extra.get("agent_events") or [])
    tool_events = [
        e for e in all_events[offset:]
        if isinstance(e, dict) and e.get("type") == "tool"
    ]

    node_timing = list(extra.get("node_timing") or [])
    summary = extra.get("node_timing_summary") if isinstance(
        extra.get("node_timing_summary"), dict
    ) else None

    turn_audit = list(extra.get("turn_audit") or [])
    if not turn_audit and cfg.audit_persist:
        turn_audit = _load_audit_from_file(
            session_id, cfg.audit_log_dir, limit=6
        )

    trace_id = ""
    req = getattr(ctx, "request", None)
    if req is not None:
        trace_id = str(getattr(req, "trace_id", "") or "")
    if not trace_id and turn_audit:
        trace_id = str(turn_audit[0].get("trace_id") or "")

    return {
        "trace_id": trace_id,
        "node_timing": node_timing,
        "node_timing_summary": summary,
        "tool_events": tool_events,
        "turn_audit": turn_audit,
    }


def format_observability_markdown(snapshot: dict[str, Any]) -> str:
    """格式化为 Markdown，追加在检索结果 / 性能面板之后。"""
    parts: list[str] = ["\n\n---\n### 📊 运行监控\n"]

    trace_id = snapshot.get("trace_id")
    if trace_id:
        parts.append(f"- **trace_id**: `{trace_id}`")

    # ── 三节点耗时 ──
    summary = snapshot.get("node_timing_summary")
    nodes = (summary or {}).get("nodes") if isinstance(summary, dict) else None
    if nodes:
        parts.append("\n#### ⏱ LangGraph 节点\n")
        parts.append("| 节点 | 耗时 | 占比 |")
        parts.append("|------|------|------|")
        for n in nodes:
            parts.append(
                f"| {n.get('node', '?')} | {_fmt_ms(n.get('duration_ms'))} "
                f"| {n.get('pct', 0)}% |"
            )
        total = summary.get("total_ms")
        if total is not None:
            parts.append(f"\n**合计** {_fmt_ms(total)}")
        slow = summary.get("slow_nodes") or []
        if slow:
            parts.append(f" · ⚠️ 慢节点: {', '.join(slow)}")
    else:
        timing = snapshot.get("node_timing") or []
        graph_nodes = [
            t for t in timing
            if isinstance(t, dict) and t.get("node") in ("prepare", "agent", "persist")
        ]
        if graph_nodes:
            parts.append("\n#### ⏱ LangGraph 节点\n")
            parts.append("| 节点 | 耗时 | 慢 |")
            parts.append("|------|------|-----|")
            for t in graph_nodes[-3:]:
                parts.append(
                    f"| {t.get('node')} | {_fmt_ms(t.get('duration_ms'))} "
                    f"| {'是' if t.get('slow') else '否'} |"
                )
        else:
            parts.append("\n#### ⏱ LangGraph 节点\n")
            parts.append("_（本轮未采集到节点 timing，请确认对话已完整结束）_")

    # ── 工具调用 ──
    tools = snapshot.get("tool_events") or []
    parts.append("\n#### 🔧 工具调用\n")
    if not tools:
        parts.append("_本轮无工具调用_")
    else:
        parts.append("| 工具 | 耗时 | 状态 |")
        parts.append("|------|------|------|")
        for t in tools:
            name = t.get("tool_name") or "?"
            err = t.get("error")
            status = "❌ 失败" if err else "✅ 成功"
            parts.append(
                f"| `{name}` | {_fmt_ms(t.get('duration_ms'))} | {status} |"
            )

    # ── 审计 ──
    audit = snapshot.get("turn_audit") or []
    parts.append("\n#### 📋 审计（本轮）\n")
    if not audit:
        parts.append(
            "_无审计记录（需 `observability.audit_persist: true` 且对话跑完 prepare/agent/persist）_"
        )
    else:
        parts.append("| 节点 | 耗时 | 错误 | 内容 hash |")
        parts.append("|------|------|------|-----------|")
        for row in audit[-3:]:
            hashes = row.get("content_hashes") or {}
            hash_bits = ", ".join(f"{k}={v[:8]}…" for k, v in hashes.items()) or "—"
            err = row.get("error")
            parts.append(
                f"| {row.get('node', '?')} | {_fmt_ms(row.get('duration_ms'))} "
                f"| {err or '—'} | {hash_bits} |"
            )

    return "\n".join(parts)
