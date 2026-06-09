"""session_search 中文匹配 + L4 prefetch 修复。"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.chat_config import ChatAgentConfig
from app.agents.context_builder import (
    is_empty_session_search_text,
    prefetch_l4_profile,
    prefetch_session_recall,
)
from agent_platform.memory.adapters.session_search_terms import extract_search_terms
from core.composition.run_context import RunContext
from core.domain.context import RequestContext


def test_extract_search_terms_chinese():
    terms = extract_search_terms("扫地机器人如何选择")
    assert "扫地机器人如何选择" in terms
    assert "扫地" in terms
    assert "机器" in terms


def test_is_empty_session_search_text():
    assert is_empty_session_search_text("")
    assert is_empty_session_search_text("No relevant messages found")
    assert not is_empty_session_search_text("user: 之前讨论过扫地机器人")


@pytest.mark.asyncio
async def test_prefetch_l4_profile_dict_facts():
    memory = MagicMock()
    memory.fetch_profile_facts = AsyncMock(
        return_value=[
            {"key": "部门", "value": "研发部", "source": "ldap"},
            {"key": "职位", "value": "工程师", "source": "hr"},
        ]
    )
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        ),
        memory=memory,
    )
    cfg = ChatAgentConfig(
        l4_profile_prefetch=True,
        enable_l4_tools=True,
        retrieval_orchestration=True,
        l4_prefetch_on_knowledge=True,
    )
    out = await prefetch_l4_profile(ctx, "扫地机器人", cfg, enabled=True)
    assert "【外部画像 L4】" in out
    assert "部门: 研发部" in out


@pytest.mark.asyncio
async def test_prefetch_session_recall_skips_empty_marker():
    memory = MagicMock()
    memory.session_search = AsyncMock(return_value="No relevant messages found")
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        ),
        memory=memory,
    )
    cfg = ChatAgentConfig(session_search_prefetch=True)
    out = await prefetch_session_recall(
        ctx, "扫地机器人如何选择", cfg, enabled=True
    )
    assert out == ""


@pytest.mark.asyncio
async def test_prefetch_session_recall_passes_use_llm_summary_for_knowledge():
    memory = MagicMock()
    memory.session_search = AsyncMock(return_value="user: 扫地机器人讨论")
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        ),
        memory=memory,
    )
    cfg = ChatAgentConfig(
        session_search_prefetch=True,
        retrieval_orchestration=True,
        knowledge_session_search=True,
        session_search_use_llm_summary=False,
    )
    out = await prefetch_session_recall(
        ctx, "扫地机器人有什么缺点", cfg, enabled=True
    )
    assert "【会话相关检索】" in out
    memory.session_search.assert_awaited_once()
    assert memory.session_search.await_args.kwargs.get("use_llm_summary") is False


@pytest.mark.asyncio
async def test_prefetch_session_recall_strips_thinking():
    memory = MagicMock()
    memory.session_search = AsyncMock(
        return_value="摘要<think>hidden</think>"
    )
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        ),
        memory=memory,
    )
    cfg = ChatAgentConfig(session_search_prefetch=True)
    out = await prefetch_session_recall(ctx, "之前说过什么", cfg, enabled=True)
    assert "redacted_thinking" not in out
    assert "摘要" in out
