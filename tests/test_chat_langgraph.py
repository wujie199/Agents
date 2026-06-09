"""LangGraph 聊天工作流（无工具）单元测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from core.domain.context import RequestContext
from core.domain.evidence import Evidence, EvidenceBundle, SourceType
from core.composition.run_context import RunContext
from core.composition.factory import FakeModelPort
from core.ports.memory import PromptMemorySnapshot

from app.agents.chat_config import ChatAgentConfig
from app.agents.chat_langgraph import (
    create_chat_langgraph_session,
    run_chat_turn_langgraph,
)
from app.runtime.adapters.langgraph.checkpointer import get_chat_checkpointer
from app.workflows.chat.graph_def import build_chat_langgraph_workflow
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
