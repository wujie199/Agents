# -*- coding: utf-8 -*-
"""ReAct Agent astream_events 解析与 tool/LLM 指标。"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage

from core.composition.run_context import RunContext
from core.ports.observability import Layer

from app.agents.observability.graph_metrics import record_llm_call, record_tool_call
from app.agents.observability.instrument import span_ctx


def _append_agent_event(ctx: Optional[RunContext], event: dict) -> None:
    if ctx is None:
        return
    extra = getattr(ctx, "extra", None)
    if not isinstance(extra, dict):
        return
    events = extra.setdefault("agent_events", [])
    if isinstance(events, list):
        events.append(event)


def process_react_stream_event(
    event: dict,
    *,
    ctx: Optional[RunContext],
    tool_starts: Dict[str, float],
    llm_starts: Dict[str, float],
) -> Optional[List[BaseMessage]]:
    """解析单条 astream_events(v2) 事件；返回更新的 messages（若有）。"""
    kind = str(event.get("event") or "")
    run_id = str(event.get("run_id") or "")
    name = str(event.get("name") or "")

    if kind == "on_tool_start":
        tool_starts[run_id] = time.perf_counter()
        return None

    if kind == "on_tool_end":
        t0 = tool_starts.pop(run_id, None)
        duration_ms = (
            round((time.perf_counter() - t0) * 1000, 2) if t0 is not None else 0.0
        )
        record_tool_call(ctx, duration_ms, tool_name=name, success=True)
        payload = {
            "type": "tool",
            "tool_name": name,
            "duration_ms": duration_ms,
            "error": False,
        }
        _append_agent_event(ctx, payload)
        return None

    if kind == "on_tool_error":
        t0 = tool_starts.pop(run_id, None)
        duration_ms = (
            round((time.perf_counter() - t0) * 1000, 2) if t0 is not None else 0.0
        )
        record_tool_call(ctx, duration_ms, tool_name=name, success=False)
        err = (event.get("data") or {}).get("error")
        payload = {
            "type": "tool",
            "tool_name": name,
            "duration_ms": duration_ms,
            "error": True,
            "error_message": str(err) if err is not None else "",
        }
        _append_agent_event(ctx, payload)
        return None

    if kind == "on_chat_model_start":
        llm_starts[run_id] = time.perf_counter()
        return None

    if kind == "on_chat_model_end":
        t0 = llm_starts.pop(run_id, None)
        duration_ms = (
            round((time.perf_counter() - t0) * 1000, 2) if t0 is not None else 0.0
        )
        output = (event.get("data") or {}).get("output")
        prompt_tokens = None
        if output is not None:
            usage = getattr(output, "usage_metadata", None) or getattr(
                output, "response_metadata", {}
            )
            if isinstance(usage, dict):
                prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
            else:
                prompt_tokens = getattr(usage, "input_tokens", None)
        record_llm_call(ctx, duration_ms, prompt_tokens=prompt_tokens)
        _append_agent_event(
            ctx,
            {"type": "llm", "duration_ms": duration_ms, "model": name},
        )
        return None

    if kind == "on_chain_stream":
        chunk = (event.get("data") or {}).get("chunk") or {}
        if isinstance(chunk, dict):
            msgs = chunk.get("messages")
            if msgs:
                return list(msgs)
        return None

    if kind == "on_chain_end":
        output = (event.get("data") or {}).get("output") or {}
        if isinstance(output, dict):
            msgs = output.get("messages")
            if msgs:
                return list(msgs)
    return None


async def stream_react_agent_with_observability(
    react_agent: Any,
    messages: List[BaseMessage],
    config: Any,
    ctx: Optional[RunContext],
) -> List[BaseMessage]:
    """astream_events(v2) 驱动 ReAct，记录 tool/LLM 指标并保留最终 messages。"""
    out_messages = list(messages)
    tool_starts: Dict[str, float] = {}
    llm_starts: Dict[str, float] = {}

    span_attrs: Dict[str, Any] = {"node": "agent"}
    if ctx is not None:
        req = getattr(ctx, "request", None)
        if req is not None:
            for key in ("tenant_id", "user_id", "session_id"):
                val = getattr(req, key, None)
                if val:
                    span_attrs[key] = str(val)

    async with span_ctx(ctx, "graph.agent.react", Layer.AGENT, span_attrs):
        async for event in react_agent.astream_events(
            {"messages": messages},
            config,
            version="v2",
        ):
            updated = process_react_stream_event(
                event,
                ctx=ctx,
                tool_starts=tool_starts,
                llm_starts=llm_starts,
            )
            if updated:
                out_messages = updated

    return out_messages
