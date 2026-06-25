"""对话 Agent：prompt 组装与 run_chat_turn（无真实 LLM/RAG）。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.domain.context import RequestContext
from core.domain.evidence import Evidence, EvidenceBundle, SourceType
from core.composition.run_context import RunContext
from core.composition.factory import FakeModelPort
from core.ports.memory import PromptMemorySnapshot, TurnRecord

from app.agents.prompts.prompt_builder import build_chat_messages, format_evidence_bundle
from app.agents.orchestration.chat_turn import run_chat_turn
from app.agents.orchestration.chat_config import ChatAgentConfig


def test_format_evidence_bundle_empty():
    bundle = EvidenceBundle(evidences=[], empty=True)
    assert format_evidence_bundle(bundle) == ""


def test_format_evidence_bundle_with_items():
    bundle = EvidenceBundle(
        evidences=[
            Evidence(
                id="1",
                content="桥梁检测要点",
                source_type=SourceType.VECTOR,
                score=0.9,
                citation="doc.pdf",
            )
        ],
        empty=False,
    )
    text = format_evidence_bundle(bundle, max_chars=2000, max_items=5)
    assert "桥梁检测" in text
    assert "仅供参考" in text
    assert "doc.pdf" in text


def test_build_chat_messages_order():
    msgs = build_chat_messages(
        memory_system="USER: 张三",
        user_message="你好",
        evidence_text="[1] score=0.5\n片段",
        history=[{"role": "user", "content": "上一轮"}],
    )
    assert msgs[0]["role"] == "system"
    assert "张三" in msgs[0]["content"]
    assert "片段" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "上一轮"}
    assert msgs[-1] == {"role": "user", "content": "你好"}


@pytest.mark.asyncio
async def test_run_chat_turn_memory_and_rag():
    request = RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id="s_chat",
        trace_id="tr1",
        channel="test",
    )

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "你好，我是助手"
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    memory = MagicMock()
    memory.compose_prompt_snapshot.return_value = PromptMemorySnapshot(
        memory_text="L1 memory",
        hash="abc",
    )
    memory.ensure_session = AsyncMock()
    memory.persist_turn = AsyncMock()
    memory.list_turns = AsyncMock(return_value=[])

    bundle = EvidenceBundle(
        evidences=[
            Evidence(
                id="e1",
                content="知识库片段",
                source_type=SourceType.VECTOR,
                score=0.8,
            )
        ],
        empty=False,
    )
    rag = MagicMock()
    rag.route_and_retrieve = AsyncMock(return_value=bundle)

    ctx = RunContext(
        request=request,
        memory=memory,
        rag=rag,
        models=FakeModelPort({"main_llm": mock_llm}),
    )

    cfg = ChatAgentConfig(
        enable_rag=True,
        max_history_turns=5,
        model_role="main_llm",
        enable_memory_tools=False,
    )
    result = await run_chat_turn(ctx, "用户问题", chat_cfg=cfg)

    assert result.assistant_text == "你好，我是助手"
    assert result.evidence_count == 1
    assert not result.rag_empty
    rag.route_and_retrieve.assert_awaited_once_with("用户问题", request, plan=None)

    mock_llm.ainvoke.assert_awaited_once()
    messages = mock_llm.ainvoke.call_args[0][0]
    assert "L1 memory" in messages[0]["content"]
    assert "知识库片段" in messages[0]["content"]

    assert memory.persist_turn.await_count == 2
    calls = [c.args[1] for c in memory.persist_turn.await_args_list]
    assert calls[0].role == "user" and calls[0].content == "用户问题"
    assert calls[1].role == "assistant"


@pytest.mark.asyncio
async def test_run_chat_turn_rag_disabled():
    request = RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id="s2",
        trace_id="tr2",
        channel="test",
    )
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "ok"
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    memory = MagicMock()
    memory.compose_prompt_snapshot.return_value = PromptMemorySnapshot(
        memory_text="mem", hash="h"
    )
    memory.ensure_session = AsyncMock()
    memory.persist_turn = AsyncMock()
    memory.list_turns = AsyncMock(return_value=[])

    rag = MagicMock()
    rag.route_and_retrieve = AsyncMock()

    ctx = RunContext(
        request=request,
        memory=memory,
        rag=rag,
        models=FakeModelPort({"main_llm": mock_llm}),
    )

    cfg = ChatAgentConfig(enable_memory_tools=False)
    await run_chat_turn(ctx, "hi", chat_cfg=cfg, enable_rag=False)
    rag.route_and_retrieve.assert_not_awaited()
