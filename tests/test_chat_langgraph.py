"""LangGraph 聊天工作流（无工具）单元测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from core.domain.context import RequestContext
from core.domain.evidence import Evidence, EvidenceBundle, SourceType
from core.composition.run_context import RunContext
from core.composition.factory import FakeModelPort
from core.ports.memory import PromptMemorySnapshot

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.orchestration.chat_langgraph import (
    create_chat_langgraph_session,
    run_chat_turn_langgraph,
)
from app.runtime.adapters.langgraph.checkpointer import get_chat_checkpointer
from agent_platform.memory.adapters.context_compressor import (
    CompressorConfig,
    HermesContextCompressor,
)
from app.agents.memory.l0_context import get_l0_state
from app.agents.roles.react_turn import dict_messages_to_lc
from app.workflows.chat.graph_def import _compress_node, build_chat_langgraph_workflow
from app.runtime.adapters.langgraph.engine import LangGraphRuntime


def _request() -> RequestContext:
    return RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id="lg_sess",
        trace_id="tr",
        channel="test",
    )


def _mock_ctx() -> RunContext:
    request = _request()
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.usage = None
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "LangGraph 回复"
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    memory = MagicMock()
    memory.compose_prompt_snapshot.return_value = PromptMemorySnapshot(
        memory_text="L1", hash="h"
    )
    memory.ensure_session = AsyncMock()
    memory.persist_turn = AsyncMock()
    memory.list_turns = AsyncMock(return_value=[])
    memory.l1_nudge_interval = 0

    bundle = EvidenceBundle(
        evidences=[
            Evidence(
                id="1",
                content="证据",
                source_type=SourceType.VECTOR,
                score=0.9,
            )
        ],
        empty=False,
    )
    rag = MagicMock()
    rag.route_and_retrieve = AsyncMock(return_value=bundle)

    return RunContext(
        request=request,
        memory=memory,
        rag=rag,
        models=FakeModelPort({"main_llm": mock_llm}),
    )


@pytest.mark.asyncio
async def test_langgraph_chat_turn():
    ctx = _mock_ctx()
    cfg = ChatAgentConfig(
        enable_rag=True,
        enable_memory_tools=False,
        model_role="main_llm",
    )
    session = create_chat_langgraph_session(
        ctx, chat_cfg=cfg, checkpointer=get_chat_checkpointer()
    )
    result = await run_chat_turn_langgraph(
        session, ctx, "介绍一下扫地机器人", enable_rag=True
    )
    assert result.assistant_text == "LangGraph 回复"
    assert result.evidence_count == 1
    assert ctx.require_memory().persist_turn.await_count == 2


@pytest.mark.asyncio
async def test_langgraph_workflow_compile_invoke():
    ctx = _mock_ctx()
    cfg = ChatAgentConfig(enable_rag=False, enable_memory_tools=False)
    workflow = build_chat_langgraph_workflow(ctx, cfg)
    runtime = LangGraphRuntime(checkpointer=get_chat_checkpointer())
    compiled = runtime.compile(workflow)
    out = await runtime.ainvoke(
        compiled,
        {"user_input": "test"},
        ctx,
        enable_rag=False,
        chat_cfg=cfg,
    )
    assert "assistant_text" in out
    assert out["assistant_text"] == "LangGraph 回复"


@pytest.mark.asyncio
async def test_compress_node_sets_l0_applied(monkeypatch):
    ctx = _mock_ctx()
    comp = HermesContextCompressor(
        config=CompressorConfig(
            trigger_pct=0.1,
            context_window_tokens=800,
            compress_target_ratio=0.2,
            anti_jitter_pct=0.01,
            enabled=True,
        )
    )

    async def _short_summary(head, context, **kwargs):
        return "[Context Summary]\nok"

    monkeypatch.setattr(comp, "_generate_summary", _short_summary)
    ctx.extra = {
        "context_compressor": comp,
        "_last_llm_prompt_tokens": 300,
    }
    cfg = ChatAgentConfig(
        enable_rag=False,
        enable_memory_tools=False,
        l0_context_compress_enabled=True,
        context_compress_threshold=0.1,
        context_window_tokens=800,
        compress_target_ratio=0.2,
    )
    long_msg = "x" * 400
    dict_msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        *[{"role": "assistant", "content": long_msg} for _ in range(8)],
        {"role": "assistant", "content": "latest"},
    ]
    state = {
        "messages": dict_messages_to_lc(dict_msgs),
        "memory_snapshot_hash": "h1",
    }
    config = {"configurable": {"run_ctx": ctx, "chat_cfg": cfg}}
    out = await _compress_node(state, config)
    assert out["l0_applied"] is True
    assert ctx.extra.get("_l0_compress_count") == 1
    assert get_l0_state(ctx) is not None


@pytest.mark.asyncio
async def test_langgraph_workflow_l0_compress_integration(monkeypatch):
    """完整图 prepare → agent → compress → persist：超阈值时 l0_applied=True。"""
    ctx = _mock_ctx()
    comp = HermesContextCompressor(
        config=CompressorConfig(
            trigger_pct=0.1,
            context_window_tokens=800,
            compress_target_ratio=0.2,
            anti_jitter_pct=0.01,
            enabled=True,
        )
    )

    async def _short_summary(head, context, **kwargs):
        return "[Context Summary]\nok"

    monkeypatch.setattr(comp, "_generate_summary", _short_summary)
    ctx.extra = {
        "context_compressor": comp,
        "_last_llm_prompt_tokens": 300,
    }
    cfg = ChatAgentConfig(
        enable_rag=False,
        enable_memory_tools=False,
        l0_context_compress_enabled=True,
        context_compress_threshold=0.1,
        context_window_tokens=800,
        compress_target_ratio=0.2,
    )
    long_msg = "x" * 400

    async def _long_turn_messages(_ctx, user_input, _chat_cfg, enable_rag=True):
        dict_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            *[{"role": "assistant", "content": long_msg} for _ in range(8)],
            {"role": "user", "content": user_input},
        ]
        return dict_messages, 0, True, "h1", [], {}

    monkeypatch.setattr(
        "app.workflows.chat.graph_def.build_turn_messages",
        _long_turn_messages,
    )
    workflow = build_chat_langgraph_workflow(ctx, cfg)
    runtime = LangGraphRuntime(checkpointer=get_chat_checkpointer())
    compiled = runtime.compile(workflow)
    out = await runtime.ainvoke(
        compiled,
        {"user_input": "继续"},
        ctx,
        enable_rag=False,
        chat_cfg=cfg,
    )
    assert out["l0_applied"] is True
    assert ctx.extra.get("_l0_compress_count") == 1
    l0 = get_l0_state(ctx)
    assert l0 is not None
    assert l0.get("memory_snapshot_hash") == "h1"
