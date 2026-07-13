"""Context Builder 单元测试。"""

from __future__ import annotations

import pytest

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.context_builder import (
    _collect_recent_user_messages,
    compress_history_for_knowledge,
    is_cross_session_recall,
    is_meta_recall_query,
    is_name_intro_query,
    is_recall_query,
    prefetch_session_recall,
    resolve_recall_scope,
    should_run_rag,
    split_history_for_summary,
    trim_history_tail_chars,
    validate_l1_key,
)


def test_should_run_rag_gates_greeting():
    cfg = ChatAgentConfig(enable_rag_gating=True)
    assert not should_run_rag("你好", cfg)
    assert not should_run_rag("hello", cfg)
    assert should_run_rag("如何选择扫地机器人", cfg)


def test_is_recall_query():
    assert is_recall_query("之前问过你什么问题")
    assert not is_recall_query("今天天气怎么样")


def test_is_meta_recall_query():
    assert is_meta_recall_query("之前问过你什么问题")
    assert is_meta_recall_query("我们聊过什么")
    assert not is_meta_recall_query("上次说的清洁方案怎么样了")
    assert not is_meta_recall_query("之前说过什么品牌")
    assert not is_meta_recall_query("如何选择扫地机器人")


def test_collect_recent_user_messages_dedupes():
    rows = [
        {"role": "user", "content": "a", "ts": "2026-07-01T10:00:00"},
        {"role": "assistant", "content": "x", "ts": "2026-07-01T10:01:00"},
        {"role": "user", "content": "b", "ts": "2026-07-02T10:00:00"},
        {"role": "user", "content": "a", "ts": "2026-07-02T11:00:00"},
    ]
    out = _collect_recent_user_messages(rows, limit=3, dedupe=True)
    assert [r["content"] for r in out] == ["b", "a"]


def test_cross_session_recall_scope():
    cfg = ChatAgentConfig(session_search_prefetch_scope="auto")
    assert is_cross_session_recall("查一下历史会话里的讨论")
    assert resolve_recall_scope("查一下历史会话里的讨论", cfg) == "user"
    assert resolve_recall_scope("之前说过什么", cfg) == "session"
    cfg_user = ChatAgentConfig(session_search_prefetch_scope="user")
    assert resolve_recall_scope("你好", cfg_user) == "user"


def test_split_history_for_summary():
    hist = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    older, recent = split_history_for_summary(hist, recent_turns=2)
    assert len(recent) == 4
    assert len(older) == 16


def test_validate_l1_key():
    allowed = ("称呼", "语言")
    assert validate_l1_key("称呼", allowed) == "称呼"
    with pytest.raises(ValueError):
        validate_l1_key("密码", allowed)


def test_trim_history_tail_chars():
    hist = [
        {"role": "user", "content": "a" * 100},
        {"role": "assistant", "content": "b" * 100},
        {"role": "user", "content": "short"},
    ]
    trimmed = trim_history_tail_chars(hist, 120)
    assert trimmed[-1]["content"] == "short"
    assert len(trimmed) <= 2


@pytest.mark.asyncio
async def test_prepare_session_context_no_llm():
    from unittest.mock import AsyncMock, MagicMock

    from core.composition.run_context import RunContext
    from core.domain.context import RequestContext
    from app.agents.context_builder import prepare_session_context

    cfg = ChatAgentConfig(
        enable_rolling_summary=True,
        use_llm_rolling_summary=False,
        session_search_prefetch=False,
        recent_turns=1,
        max_history_chars=500,
    )
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t",
            user_id="u",
            session_id="s",
            trace_id="tr",
            channel="test",
        )
    )
    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new q"},
        {"role": "assistant", "content": "new a"},
    ]
    result = await prepare_session_context(ctx, history, "new q", cfg)
    assert len(result.trimmed_history) <= 2
    assert "更早对话摘要" in result.extra_text


@pytest.mark.asyncio
async def test_prefetch_session_recall_meta_uses_recent_turns():
    from unittest.mock import AsyncMock, MagicMock

    from core.composition.run_context import RunContext
    from core.domain.context import RequestContext

    memory = MagicMock()
    memory.session_search = AsyncMock(return_value="semantic hit")
    memory.list_recent_turns = AsyncMock(
        return_value=[
            {"role": "user", "content": "清洁保养机身应注意什么", "ts": "2026-07-02T23:39:00"},
            {"role": "user", "content": "之前问过你什么问题", "ts": "2026-06-04T22:44:00"},
        ]
    )
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t",
            user_id="u",
            session_id="s",
            trace_id="tr",
            channel="test",
        ),
        memory=memory,
    )
    cfg = ChatAgentConfig(
        session_search_prefetch=True,
        meta_recall_recent_list=True,
        meta_recall_recent_limit=5,
    )
    out = await prefetch_session_recall(ctx, "之前问过什么", cfg, enabled=True)
    assert "会话回忆·最近提问" in out
    assert "清洁保养" in out
    assert "semantic hit" not in out
    memory.list_recent_turns.assert_awaited_once()
    memory.session_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_prefetch_session_recall_topic_uses_semantic_search():
    from unittest.mock import AsyncMock, MagicMock

    from core.composition.run_context import RunContext
    from core.domain.context import RequestContext

    memory = MagicMock()
    memory.session_search = AsyncMock(return_value="之前讨论过扫地机器人")
    memory.list_recent_turns = AsyncMock(return_value=[])
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t",
            user_id="u",
            session_id="s",
            trace_id="tr",
            channel="test",
        ),
        memory=memory,
    )
    cfg = ChatAgentConfig(session_search_prefetch=True, meta_recall_recent_list=True)
    out = await prefetch_session_recall(
        ctx, "上次说的清洁方案怎么样了", cfg, enabled=True
    )
    assert "会话回忆" in out
    assert "扫地机器人" in out
    memory.session_search.assert_awaited_once()
    memory.list_recent_turns.assert_not_awaited()


def test_is_name_intro_query():
    assert is_name_intro_query("叫武杰")
    assert is_name_intro_query("我叫李明")
    assert not is_name_intro_query("你好")


def test_compress_history_for_knowledge():
    hist = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a" * 300},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "b" * 300},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "c" * 50},
    ]
    out = compress_history_for_knowledge(hist, keep_assistant_turns=2)
    assert [r["content"] for r in out if r["role"] == "user"] == ["u1", "u2", "u3"]
    assistants = [r["content"] for r in out if r["role"] == "assistant"]
    assert len(assistants) == 2
    assert all(len(a) <= 181 for a in assistants)
