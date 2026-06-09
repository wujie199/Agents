"""企业级记忆管理单元测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.domain.context import RequestContext
from core.composition.run_context import RunContext
from core.ports.memory import PromptMemorySnapshot

from app.agents.enterprise_memory import (
    confirm_pending_l1,
    get_memory_status,
    list_user_sessions,
    memory_config_summary,
)


def test_memory_config_summary():
    summary = memory_config_summary()
    assert "archive_backend" in summary
    assert "retention_days" in summary


@pytest.mark.asyncio
async def test_list_user_sessions():
    memory = MagicMock()
    memory.list_sessions = AsyncMock(
        return_value=[{"session_id": "s1", "status": "closed"}]
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
    rows = await list_user_sessions(ctx, limit=10)
    assert rows[0]["session_id"] == "s1"


@pytest.mark.asyncio
async def test_get_memory_status():
    memory = MagicMock()
    memory.compose_prompt_snapshot.return_value = PromptMemorySnapshot(
        memory_text="L1", hash="abc"
    )
    memory.list_sessions = AsyncMock(return_value=[])
    memory._hot = None
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
    status = await get_memory_status(ctx)
    assert status["l1_hash"] == "abc"
    assert "config" in status


@pytest.mark.asyncio
async def test_confirm_pending_l1():
    memory = MagicMock()
    memory.confirm_pending_deltas = AsyncMock(return_value=2)
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
    n = await confirm_pending_l1(ctx)
    assert n == 2
