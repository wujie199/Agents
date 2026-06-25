"""次要问题修复：历史去重、直连 LLM 路由、session_search user 回退。"""

from __future__ import annotations

import pytest

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.context_builder import (
    compress_history_for_knowledge,
    dedupe_history_turns,
)
from app.agents.roles.retrieval_router import should_use_direct_llm_for_intent


def test_dedupe_history_turns_consecutive():
    hist = [
        {"role": "user", "content": "你好"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "hi"},
    ]
    out = dedupe_history_turns(hist)
    assert len(out) == 2
    assert out[0]["content"] == "你好"


def test_dedupe_history_turns_collapse_user():
    hist = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "如何选取扫地机器人"},
        {"role": "user", "content": "你好"},
    ]
    out = dedupe_history_turns(hist, collapse_duplicate_user=True)
    users = [r["content"] for r in out if r["role"] == "user"]
    assert users == ["如何选取扫地机器人", "你好"]


def test_compress_history_dedupes_user():
    hist = [
        {"role": "user", "content": "你好"},
        {"role": "user", "content": "你好"},
        {"role": "user", "content": "q"},
    ]
    out = compress_history_for_knowledge(hist)
    users = [r["content"] for r in out if r["role"] == "user"]
    assert users == ["你好", "q"]
    assert users.count("你好") == 1


def test_should_use_direct_llm_for_intent():
    cfg = ChatAgentConfig(enable_memory_tools=True, knowledge_direct_llm=True)
    assert should_use_direct_llm_for_intent("knowledge", cfg)
    assert should_use_direct_llm_for_intent("chitchat", cfg)
    assert not should_use_direct_llm_for_intent("recall", cfg)
    assert not should_use_direct_llm_for_intent("profile", cfg)
    off = ChatAgentConfig(enable_memory_tools=True, knowledge_direct_llm=False)
    assert not should_use_direct_llm_for_intent("knowledge", off)


@pytest.mark.asyncio
async def test_session_search_user_fallback():
    from unittest.mock import AsyncMock, MagicMock

    from core.composition.run_context import RunContext
    from core.domain.context import RequestContext

    from agent_platform.memory.adapters.memory_port_adapter import MemoryPortAdapter

    archive = MagicMock()
    archive.search_messages = AsyncMock(
        return_value=[
            {
                "message_id": "1",
                "session_id": "s",
                "role": "assistant",
                "content": "扫地机器人很好",
                "ts": "t1",
            }
        ]
    )
    archive.select_many = AsyncMock(
        return_value=[
            {
                "message_id": "2",
                "session_id": "s",
                "role": "user",
                "content": "扫地机器人的好处",
                "ts": "t2",
            }
        ]
    )
    adapter = MemoryPortAdapter(
        archive_db=archive,
        hot_memory=MagicMock(),
        session_hybrid_search=False,
        session_search_rerank=False,
    )
    adapter._ensure_session_vector_index_version = AsyncMock()
    adapter._filter_messages_by_acl = lambda msgs, _ctx: msgs
    ctx = RequestContext(
        tenant_id="t", user_id="u", session_id="s", trace_id="tr", channel="test"
    )
    detail = await adapter.session_search_detail(
        "扫地机器人的好处",
        ctx,
        limit=3,
        prefer_user_role=True,
        use_llm_summary=False,
    )
    assert detail.fragments
    assert all(f.role == "user" for f in detail.fragments)
