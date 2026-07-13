# -*- coding: utf-8 -*-
"""LangGraph Hermes 图状态单元测试。"""

from __future__ import annotations

from core.composition.run_context import RunContext
from core.domain.context import RequestContext

from app.agents.memory.memory_graph_state import (
    append_pending_memory_delta,
    build_prepare_state_patch,
    clear_pending_memory_delta,
    pending_memory_delta_from_ctx,
    resolve_memory_path,
    working_memory_from_ctx,
)
from app.agents.orchestration.chat_config import ChatAgentConfig
from core.ports.memory import MemoryDelta


def _ctx(**extra) -> RunContext:
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        )
    )
    ctx.extra = dict(extra)
    return ctx


def test_resolve_memory_path():
    assert resolve_memory_path(ChatAgentConfig(enable_memory_tools=False)) == "path_a"
    assert resolve_memory_path(ChatAgentConfig(enable_memory_tools=True)) == "path_b"


def test_working_memory_from_ctx():
    wm = working_memory_from_ctx(
        _ctx(retrieval_intent="knowledge", pending_remember="姓名=张三")
    )
    assert wm["retrieval_intent"] == "knowledge"
    assert wm["pending_remember"] == "姓名=张三"


def test_build_prepare_state_patch_includes_working_memory():
    ctx = _ctx(retrieval_intent="maintenance", pending_remember="x")
    cfg = ChatAgentConfig(enable_memory_tools=True)
    patch = build_prepare_state_patch(
        user_input="hi",
        lc_messages=[],
        ev_count=2,
        rag_empty=False,
        mem_hash="abc",
        evidences_summary=[{"id": "1"}],
        memory_summary={"recall_hit": True},
        ctx=ctx,
        chat_cfg=cfg,
    )
    assert patch["memory_path"] == "path_b"
    assert patch["retrieval_intent"] == "maintenance"
    assert patch["pending_remember"] == "x"
    assert patch["l0_applied"] is False
    assert patch["memory_snapshot_hash"] == "abc"


def test_pending_memory_delta_accumulation():
    ctx = _ctx()
    delta = MemoryDelta(key="语言", value="中文", source="user")
    updated = append_pending_memory_delta(ctx, delta, require_hitl=True)
    assert len(updated) == 1
    assert updated[0]["key"] == "语言"
    assert ctx.extra["pending_remember"] == "语言=中文"
    append_pending_memory_delta(
        ctx, MemoryDelta(key="语言", value="英文", source="user"), require_hitl=True
    )
    assert len(pending_memory_delta_from_ctx(ctx)) == 1
    assert pending_memory_delta_from_ctx(ctx)[0]["value"] == "英文"
    clear_pending_memory_delta(ctx)
    assert pending_memory_delta_from_ctx(ctx) == []
