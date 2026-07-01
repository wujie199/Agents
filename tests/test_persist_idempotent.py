"""persist 幂等：direct 流式 + LangGraph persist 节点不重复写入。"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.orchestration.chat_nodes import persist_user_and_assistant
from core.composition.run_context import RunContext
from core.domain.context import RequestContext


@pytest.mark.asyncio
async def test_persist_user_and_assistant_skips_second_call_same_turn():
    memory = MagicMock()
    memory.persist_turn = AsyncMock()
    memory.list_turns = AsyncMock(return_value=[])

    req = RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        trace_id="tr",
        channel="test",
    )
    ctx = RunContext(
        request=req,
        memory=memory,
        extra={"_active_turn_id": "turn-abc"},
    )

    await persist_user_and_assistant(
        ctx,
        user_message="你好",
        assistant_text="你好呀",
    )
    assert memory.persist_turn.await_count == 2
    assert ctx.extra["_turn_persisted_id"] == "turn-abc"

    await persist_user_and_assistant(
        ctx,
        user_message="你好",
        assistant_text="又一次",
    )
    assert memory.persist_turn.await_count == 2
