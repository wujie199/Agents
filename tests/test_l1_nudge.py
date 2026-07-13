# -*- coding: utf-8
"""L1 nudge 周期抽取测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.composition.run_context import RunContext
from core.domain.context import RequestContext

from app.agents.memory.l1_nudge import maybe_nudge_memory_review
from app.agents.memory.memory_graph_state import pending_memory_delta_from_ctx
from app.agents.orchestration.chat_config import ChatAgentConfig


@pytest.mark.asyncio
async def test_nudge_writes_pending_when_facts_found():
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        )
    )
    ctx.extra = {}
    ctx.models = MagicMock()
    memory = MagicMock()
    memory.l1_nudge_interval = 10
    memory.update_prompt_memory = AsyncMock()
    ctx.memory = memory

    cfg = ChatAgentConfig(
        enable_memory_tools=True,
        enable_l1_extract_on_finalize=True,
        use_llm_l1_extract=True,
        remember_require_hitl=True,
        l1_auto_write_confidence_min=0.6,
    )

    import app.agents.memory.l1_nudge as nudge_mod

    async def _fake_history(*a, **k):
        return [{"role": "user", "content": "叫我张三"}]

    async def _fake_extract(ctx, turns, chat_cfg):
        return [{"key": "称呼", "value": "张三", "confidence": "0.95"}]

    nudge_mod.fetch_turn_history = _fake_history
    nudge_mod.extract_l1_facts_from_session = _fake_extract
    try:
        for _ in range(9):
            await maybe_nudge_memory_review(ctx, nudge_interval=10, chat_cfg=cfg)
        result = await maybe_nudge_memory_review(ctx, nudge_interval=10, chat_cfg=cfg)
    finally:
        import app.agents.orchestration.chat_nodes as cn

        nudge_mod.fetch_turn_history = cn.fetch_turn_history
        import app.agents.context_builder as cb

        nudge_mod.extract_l1_facts_from_session = cb.extract_l1_facts_from_session

    assert result and result.get("facts_written", 0) >= 1
    assert pending_memory_delta_from_ctx(ctx)
