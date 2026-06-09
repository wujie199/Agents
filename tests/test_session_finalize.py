"""会话结束 L2→L1 抽取单元测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.composition.run_context import RunContext
from core.domain.context import RequestContext

from app.agents.chat_config import ChatAgentConfig
from app.agents.session_finalize import enrich_l1_before_finalize


def _ctx() -> RunContext:
    memory = MagicMock()
    memory.update_prompt_memory = AsyncMock()
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


@pytest.mark.asyncio
async def test_enrich_l1_skipped_when_disabled():
    ctx = _ctx()
    n = await enrich_l1_before_finalize(
        ctx, ChatAgentConfig(enable_l1_extract_on_finalize=False)
    )
    assert n == 0
    ctx.require_memory().update_prompt_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_enrich_l1_writes_extracted_facts():
    ctx = _ctx()
    cfg = ChatAgentConfig(
        enable_l1_extract_on_finalize=True,
        remember_require_hitl=True,
    )
    turns = [
        {"role": "user", "content": "以后叫我老王"},
        {"role": "assistant", "content": "好的"},
    ]
    with patch(
        "app.agents.session_finalize.fetch_turn_history",
        new=AsyncMock(return_value=turns),
    ), patch(
        "app.agents.session_finalize.extract_l1_facts_from_session",
        new=AsyncMock(return_value=[{"key": "称呼", "value": "老王"}]),
    ):
        n = await enrich_l1_before_finalize(ctx, cfg)
    assert n == 1
    ctx.require_memory().update_prompt_memory.assert_awaited_once()


@pytest.mark.asyncio
async def test_enrich_l1_filters_invalid_keys():
    ctx = _ctx()
    cfg = ChatAgentConfig(enable_l1_extract_on_finalize=True)
    with patch(
        "app.agents.session_finalize.fetch_turn_history",
        new=AsyncMock(return_value=[{"role": "user", "content": "x"}]),
    ), patch(
        "app.agents.session_finalize.extract_l1_facts_from_session",
        new=AsyncMock(return_value=[{"key": "密码", "value": "123"}]),
    ):
        n = await enrich_l1_before_finalize(ctx, cfg)
    assert n == 0
