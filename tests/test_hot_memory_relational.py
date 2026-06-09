"""L1 relational 热记忆单元测试。"""

from __future__ import annotations

import pytest

from agent_platform.storage.adapters.sqlite.relational_adapter import (
    AsyncSQLiteRelationalAdapter,
)
from agent_platform.memory.adapters.hot_memory_relational_adapter import (
    HotMemoryRelationalAdapter,
)
from core.domain.context import RequestContext
from core.ports.memory import MemoryDelta


@pytest.mark.asyncio
async def test_hot_memory_relational_roundtrip(tmp_path):
    db = AsyncSQLiteRelationalAdapter(
        db_path=str(tmp_path / "archive.db"), pool_size=2
    )
    hot = HotMemoryRelationalAdapter(db)
    ctx = RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        trace_id="tr",
        channel="test",
    )
    hot.apply_delta("t1", "u1", MemoryDelta(key="称呼", value="小明", source="user"))
    snap = hot.compose_snapshot(ctx)
    assert "小明" in snap.memory_text
    assert snap.hash

    hot.queue_pending_delta(
        "t1", "u1", MemoryDelta(key="语言", value="中文", source="user")
    )
    pending = hot.list_pending_deltas("t1", "u1")
    assert len(pending) == 1
    flushed = hot.flush_pending_deltas("t1", "u1")
    assert len(flushed) == 1
    assert hot.list_pending_deltas("t1", "u1") == []
