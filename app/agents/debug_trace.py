# -*- coding: utf-8 -*-
"""Agent 记忆/RAG 调试追踪（--debug 时写 NDJSON + 可选控制台）。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_AGENT_LOG = Path(
    os.environ.get(
        "AGENT_DEBUG_LOG",
        str(Path(__file__).resolve().parents[2] / ".cursor" / "agent-debug.log"),
    )
)
_SESSION_ID = os.environ.get("AGENT_DEBUG_SESSION", "agent")
_DEBUG_ENABLED = os.environ.get("AGENT_DEBUG", "").lower() in ("1", "true", "yes")
_CONSOLE = True
_MIRROR_UNIFIED = True


def agent_debug_log_path() -> Path:
    return _AGENT_LOG


def set_debug_console(enabled: bool) -> None:
    global _DEBUG_ENABLED, _CONSOLE
    _DEBUG_ENABLED = enabled
    _CONSOLE = enabled


def set_debug_quiet(enabled: bool) -> None:
    """开启调试日志但不在控制台打印 agent_debug 摘要。"""
    global _DEBUG_ENABLED, _CONSOLE
    _DEBUG_ENABLED = enabled
    _CONSOLE = False


def agent_debug(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
    *,
    console: bool | None = None,
    run_id: str = "default",
) -> None:
    if not _DEBUG_ENABLED:
        return
    payload = {
        "sessionId": _SESSION_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
        "runId": run_id,
    }
    try:
        _AGENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _AGENT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass

    if _MIRROR_UNIFIED:
        try:
            from app.agents.memory_runtime_debug import is_memory_runtime_debug, trace_write

            if is_memory_runtime_debug():
                trace_write(
                    hypothesis_id=hypothesis_id,
                    location=location,
                    message=message,
                    data=data or {},
                    run_id=run_id,
                )
        except Exception:
            pass

    show = console if console is not None else _CONSOLE
    if show:
        preview = json.dumps(data or {}, ensure_ascii=False)
        if len(preview) > 1200:
            preview = preview[:1200] + "…"
        print(f"\n[debug:{hypothesis_id}] {message}\n  {preview}\n")


def summarize_messages(messages: list[dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(messages), "roles": []}
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content") or ""
        out["roles"].append(
            {
                "role": role,
                "chars": len(content),
                "preview": content[:300].replace("\n", "\\n"),
            }
        )
    return out
