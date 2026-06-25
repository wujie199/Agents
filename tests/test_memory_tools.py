"""Path B 记忆工具单元测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.composition.run_context import RunContext
from core.domain.context import RequestContext

from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.memory.memory_tools import build_memory_tools


def _ctx() -> RunContext:
    memory = MagicMock()
    memory.session_search = AsyncMock(return_value="历史片段")
    memory.update_prompt_memory = AsyncMock()
    memory.skill_search = AsyncMock(return_value=[])
    memory.run_skill = AsyncMock()
    memory.resolve_entity = AsyncMock(return_value=None)
    memory.fetch_profile_facts = AsyncMock(return_value=[])
    return RunContext(
        request=RequestContext(
            tenant_id="t",
            user_id="u",
            session_id="s",
            trace_id="tr",
            channel="test",
        ),
        memory=memory,
    )


def test_build_memory_tools_disabled():
    tools = build_memory_tools(_ctx(), ChatAgentConfig(enable_memory_tools=False))
    assert tools == []


def test_build_memory_tools_full_stack():
    tools = build_memory_tools(
        _ctx(),
        ChatAgentConfig(
            enable_memory_tools=True,
            enable_skill_tools=True,
            enable_l4_tools=True,
        ),
    )
    names = {t.name for t in tools}
    assert names == {
        "session_search",
        "remember_user_fact",
        "skill_search",
        "run_skill",
        "resolve_entity",
        "fetch_profile_facts",
    }


def test_build_memory_tools_l2_only():
    tools = build_memory_tools(
        _ctx(),
        ChatAgentConfig(
            enable_memory_tools=True,
            enable_skill_tools=False,
            enable_l4_tools=False,
        ),
    )
    names = {t.name for t in tools}
    assert names == {"session_search", "remember_user_fact"}


@pytest.mark.asyncio
async def test_session_search_tool_user_scope():
    ctx = _ctx()
    tools = build_memory_tools(ctx, ChatAgentConfig(enable_memory_tools=True))
    by_name = {t.name: t for t in tools}
    await by_name["session_search"].ainvoke(
        {"query": "扫地机器人", "limit": 3, "scope": "user"}
    )
    ctx.require_memory().session_search.assert_awaited_once()
    call_kw = ctx.require_memory().session_search.await_args.kwargs
    assert call_kw.get("scope") == "user"


@pytest.mark.asyncio
async def test_remember_user_fact_allowed_key():
    ctx = _ctx()
    cfg = ChatAgentConfig(enable_memory_tools=True, remember_require_hitl=True)
    tools = build_memory_tools(ctx, cfg)
    by_name = {t.name: t for t in tools}
    out = await by_name["remember_user_fact"].ainvoke(
        {"key": "称呼", "value": "小明"}
    )
    assert "待确认" in out
    ctx.require_memory().update_prompt_memory.assert_awaited_once()


@pytest.mark.asyncio
async def test_remember_user_fact_rejects_unknown_key():
    ctx = _ctx()
    tools = build_memory_tools(ctx, ChatAgentConfig(enable_memory_tools=True))
    by_name = {t.name: t for t in tools}
    with pytest.raises(Exception):
        await by_name["remember_user_fact"].ainvoke(
            {"key": "密码", "value": "secret"}
        )
