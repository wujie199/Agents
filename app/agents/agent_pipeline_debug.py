# -*- coding: utf-8 -*-
"""Agent 全链路组件状态 NDJSON（--debug / AGENT_PIPELINE_DEBUG=1 时写入）。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_LOG = Path(
    os.environ.get(
        "MEMORY_DEBUG_LOG",
        str(Path(__file__).resolve().parents[2] / ".cursor" / "memory_runtime.ndjson"),
    )
)
_SESSION = os.environ.get("MEMORY_DEBUG_SESSION", "default")


def pipeline_debug_enabled() -> bool:
    env = os.environ.get("AGENT_PIPELINE_DEBUG", "0").lower()
    return env in ("1", "true", "yes", "on")


def debug_log_path() -> Path:
    return _LOG


def pipeline_debug_log(
    *,
    component: str,
    stage: str,
    status: str,
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
    run_id: str = "default",
    hypothesis_id: str | None = None,
) -> None:
    """写入组件运行状态（component/stage/status 便于按层过滤）。"""
    if not pipeline_debug_enabled():
        return
    payload = {
        "sessionId": _SESSION,
        "hypothesisId": hypothesis_id or f"{component}-{status}",
        "component": component,
        "stage": stage,
        "status": status,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
        "runId": run_id,
    }
    try:
        _LOG.parent.mkdir(parents=True, exist_ok=True)
        with _LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass


def log_turn_pipeline_summary(
    ctx: Any,
    *,
    user_message: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    """一轮 build_turn 结束：汇总各层 trigger + 决策。"""
    run_id = "default"
    triggers: list[dict[str, Any]] = []
    if ctx is not None:
        run_id = getattr(getattr(ctx, "request", None), "session_id", None) or run_id
        triggers = list((getattr(ctx, "extra", None) or {}).get("layer_triggers") or [])
    summary = {
        "user_preview": (user_message or "")[:120],
        "trigger_count": len(triggers),
        "triggered": [
            f"{t.get('layer')}:{t.get('action')}"
            for t in triggers
            if t.get("triggered")
        ],
        "skipped": [
            f"{t.get('layer')}:{t.get('action')}({t.get('reason')})"
            for t in triggers
            if not t.get("triggered")
        ],
        "timeline": triggers,
    }
    if extra:
        summary.update(extra)
    decision = (getattr(ctx, "extra", None) or {}).get("turn_decision") if ctx else None
    if isinstance(decision, dict):
        summary["turn_decision"] = decision
    perf = (getattr(ctx, "extra", None) or {}).get("turn_perf") if ctx else None
    if isinstance(perf, dict):
        stages = list(perf.get("stages") or [])
        ranked = sorted(
            stages,
            key=lambda s: float(s.get("duration_ms") or 0),
            reverse=True,
        )[:5]
        summary["perf_top_ms"] = ranked
    pipeline_debug_log(
        component="TURN",
        stage="pipeline_summary",
        status="OK",
        location="agent_pipeline.turn_summary",
        message="本轮 Agent 组件运行汇总",
        data=summary,
        run_id=run_id,
        hypothesis_id="TURN-SUMMARY",
    )
