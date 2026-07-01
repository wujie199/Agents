"""检索编排 router 测试。"""

from __future__ import annotations

import pytest

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.roles.retrieval_router import build_retrieval_plan, classify_intent


def _cfg(**overrides):
    base = ChatAgentConfig(
        retrieval_orchestration=True,
        recall_wins_over_rag=True,
        knowledge_skips_recall=True,
    )
    if not overrides:
        return base
    from dataclasses import replace

    return replace(base, **overrides)


def test_classify_knowledge():
    assert classify_intent("扫地机器人有哪些品牌", _cfg()) == "knowledge"


def test_classify_recall():
    assert classify_intent("还记得我上次说的吗", _cfg()) == "recall"


def test_classify_mixed_recall_knowledge():
    """混合意图：回忆+知识倾向 → recall_and_knowledge。"""
    assert classify_intent("之前我们聊过什么", _cfg()) == "recall_and_knowledge"


def test_classify_chitchat():
    assert classify_intent("你好", _cfg()) == "chitchat"


def test_classify_profile_name():
    assert classify_intent("叫什么名字", _cfg()) == "profile"
    assert classify_intent("你知道我的哪些信息", _cfg()) == "profile"


def test_profile_skips_rag_and_session_search():
    plan = build_retrieval_plan("叫什么名字", _cfg(), enable_rag=True)
    assert plan.intent == "profile"
    assert plan.run_l4
    assert not plan.run_rag
    assert not plan.run_session_search
    assert plan.skip_rag_reason == "profile_intent"
    assert plan.skip_recall_reason == "profile_intent"


def test_profile_name_intro_short():
    plan = build_retrieval_plan("叫武杰", _cfg(), enable_rag=True)
    assert plan.intent == "profile"
    assert not plan.run_rag
    assert not plan.run_session_search


def test_recall_skips_rag():
    plan = build_retrieval_plan("还记得上次说的吗", _cfg(), enable_rag=True)
    assert plan.intent == "recall"
    assert plan.run_session_search
    assert not plan.run_rag
    assert plan.skip_rag_reason == "recall_intent"


def test_mixed_recall_knowledge_runs_rag_and_session_search():
    """混合意图同时走 session_search 和 RAG。"""
    plan = build_retrieval_plan("之前说过什么品牌", _cfg(), enable_rag=True)
    assert plan.intent == "recall_and_knowledge"
    assert plan.run_session_search
    assert plan.run_rag


def test_knowledge_skips_session_search():
    plan = build_retrieval_plan(
        "你好,请你帮我查找一下当前扫地机器人的常用品牌",
        _cfg(knowledge_session_search=False),
        enable_rag=True,
    )
    assert plan.intent == "knowledge"
    assert plan.run_rag
    assert not plan.run_session_search
    assert plan.skip_recall_reason == "knowledge_intent"


def test_knowledge_with_session_search_parallel():
    plan = build_retrieval_plan(
        "如何选择扫地机器人",
        _cfg(knowledge_session_search=True, l4_prefetch_on_knowledge=True),
        enable_rag=True,
    )
    assert plan.intent == "knowledge"
    assert plan.run_rag
    assert plan.run_session_search
    assert not plan.run_l4
    assert plan.skip_recall_reason is None


def test_knowledge_l4_when_profile_query():
    plan = build_retrieval_plan(
        "我叫什么名字",
        _cfg(knowledge_session_search=True, l4_prefetch_on_knowledge=True),
        enable_rag=True,
    )
    assert plan.intent == "profile"
    assert plan.run_l4


def test_chitchat_runs_neither():
    plan = build_retrieval_plan("你好", _cfg(), enable_rag=True)
    assert plan.intent == "chitchat"
    assert not plan.run_rag
    assert not plan.run_session_search


def test_knowledge_session_search_does_not_skip_rag_on_miss():
    """知识类并行 L2 预检索未命中时仍应走 RAG。"""
    import asyncio

    asyncio.run(_knowledge_session_search_rag_not_skipped())


async def _knowledge_session_search_rag_not_skipped():
    from unittest.mock import AsyncMock, MagicMock

    from app.agents.orchestration.chat_nodes import build_turn_messages
    from core.composition.run_context import RunContext
    from core.domain.context import RequestContext
    from core.domain.evidence import Evidence, EvidenceBundle, SourceType

    req = RequestContext(
        tenant_id="t1", user_id="u1", session_id="s1", trace_id="t", channel="test"
    )
    memory = MagicMock()
    memory.compose_prompt_snapshot.return_value = MagicMock(
        memory_text="L1", hash="h"
    )
    memory.ensure_session = AsyncMock()
    memory.list_turns = AsyncMock(return_value=[])
    memory.session_search = AsyncMock(return_value="")
    memory.fetch_profile_facts = AsyncMock(return_value=[])

    bundle = EvidenceBundle(
        evidences=[
            Evidence(id="e1", content="片段", source_type=SourceType.VECTOR, score=0.9)
        ],
        empty=False,
    )
    rag = MagicMock()
    rag.route_and_retrieve = AsyncMock(return_value=bundle)
    ctx = RunContext(request=req, memory=memory, rag=rag)

    cfg = _cfg(knowledge_session_search=True, l4_prefetch_on_knowledge=False)
    _msgs, ev_count, rag_empty, _, _, _ = await build_turn_messages(
        ctx, "扫地机器人品牌", cfg, enable_rag=True
    )
    assert ev_count == 1
    assert not rag_empty
    rag.route_and_retrieve.assert_awaited_once()


def test_legacy_mode_unchanged_recall_and_rag_independent():
    cfg = ChatAgentConfig(
        retrieval_orchestration=False,
        session_search_prefetch=True,
        enable_rag=True,
    )
    # 非回忆、非寒暄 → 仅 RAG（旧行为）
    plan = build_retrieval_plan("扫地机器人品牌", cfg, enable_rag=True)
    assert plan.intent == "legacy"
    assert plan.run_rag
    assert not plan.run_session_search
