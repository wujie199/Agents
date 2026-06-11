"""姓名自述 HITL 自动 pending。"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core.domain.context import RequestContext
from core.composition.run_context import RunContext

from app.agents.chat_config import ChatAgentConfig
from app.agents.name_remember import auto_remember_name_intro, parse_name_from_intro
from app.agents.retrieval_router import classify_intent


def _ctx() -> RunContext:
    req = RequestContext(
        tenant_id="tenant1",
        user_id="user1",
        session_id="chat1",
        trace_id="t1",
        channel="test",
    )
    memory = AsyncMock()
    memory.update_prompt_memory = AsyncMock()
    return RunContext(request=req, memory=memory)


@pytest.mark.parametrize(
    "query,expected",
    [
        ("叫武杰", "武杰"),
        ("我叫李明", "李明"),
        ("我的名字是张三", "张三"),
        ("我是王芳", "王芳"),
        ("你好", None),
        ("我是程序员", None),
    ],
)
def test_parse_name_from_intro(query, expected):
    assert parse_name_from_intro(query) == expected


@pytest.mark.asyncio
async def test_auto_remember_name_intro_pending():
    ctx = _ctx()
    cfg = ChatAgentConfig(remember_require_hitl=True)
    out = await auto_remember_name_intro(
        ctx, "叫武杰", cfg, intent="profile"
    )
    assert out == "姓名=武杰"
    ctx.memory.update_prompt_memory.assert_awaited_once()
    call_kw = ctx.memory.update_prompt_memory.await_args.kwargs
    assert call_kw.get("require_hitl") is True


@pytest.mark.asyncio
async def test_auto_remember_skips_non_profile_intent():
    ctx = _ctx()
    cfg = ChatAgentConfig()
    out = await auto_remember_name_intro(
        ctx, "叫武杰", cfg, intent="knowledge"
    )
    assert out is None
    ctx.memory.update_prompt_memory.assert_not_awaited()


def test_scenario_name_intro_intent():
    cfg = ChatAgentConfig()
    assert classify_intent("叫武杰", cfg) == "profile"
