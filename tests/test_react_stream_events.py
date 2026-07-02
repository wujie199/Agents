# -*- coding: utf-8 -*-
"""ReAct astream_events 解析单元测试。"""

from __future__ import annotations

from unittest.mock import patch

from core.domain.context import RequestContext
from core.composition.run_context import RunContext

from agent_platform.infrastructure.observability.adapter import (
    ObservabilityPortAdapter,
)

from app.agents.observability.react_events import process_react_stream_event


def _ctx() -> RunContext:
    obs = ObservabilityPortAdapter(service_name="react-events-test")
    return RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        ),
        observability=obs,
    )


def test_process_tool_end_records_metrics_and_extra():
    ctx = _ctx()
    tool_starts = {"run-1": 100.0}
    with patch(
        "app.agents.observability.react_events.time.perf_counter",
        return_value=100.05,
    ):
        result = process_react_stream_event(
            {"event": "on_tool_end", "run_id": "run-1", "name": "session_search"},
            ctx=ctx,
            tool_starts=tool_starts,
            llm_starts={},
        )
    assert result is None
    metrics = ctx.observability.get_metrics()
    assert metrics["agent.tool.calls_total"] == [1.0]
    assert metrics["agent.tool.duration_ms"][0] == 50.0
    events = ctx.extra.get("agent_events") or []
    assert events[-1]["tool_name"] == "session_search"
    assert events[-1]["error"] is False


def test_process_tool_error_marks_failure():
    ctx = _ctx()
    tool_starts = {"run-2": 10.0}
    with patch(
        "app.agents.observability.react_events.time.perf_counter",
        return_value=10.02,
    ):
        process_react_stream_event(
            {
                "event": "on_tool_error",
                "run_id": "run-2",
                "name": "memory",
                "data": {"error": ValueError("boom")},
            },
            ctx=ctx,
            tool_starts=tool_starts,
            llm_starts={},
        )
    tags = ctx.observability.get_metrics()["agent.tool.calls_total"]
    assert tags == [1.0]
    events = ctx.extra["agent_events"]
    assert events[-1]["error"] is True
    assert "boom" in events[-1]["error_message"]


def test_process_llm_end_records_duration():
    ctx = _ctx()
    llm_starts = {"llm-1": 1.0}
    with patch(
        "app.agents.observability.react_events.time.perf_counter",
        return_value=1.25,
    ):
        process_react_stream_event(
            {"event": "on_chat_model_end", "run_id": "llm-1", "name": "main_llm"},
            ctx=ctx,
            tool_starts={},
            llm_starts=llm_starts,
        )
    assert ctx.observability.get_metrics()["agent.llm.duration_ms"] == [250.0]


def test_process_chain_stream_updates_messages():
    ctx = _ctx()
    fake_msgs = [{"role": "assistant", "content": "hi"}]
    updated = process_react_stream_event(
        {
            "event": "on_chain_stream",
            "data": {"chunk": {"messages": fake_msgs}},
        },
        ctx=ctx,
        tool_starts={},
        llm_starts={},
    )
    assert updated == fake_msgs
