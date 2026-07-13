# -*- coding: utf-8
"""企业级记忆闭环：pending delta → finalize → L1。"""

from __future__ import annotations

import pytest

from core.composition.run_context import RunContext
from core.domain.context import RequestContext
from core.ports.memory import MemoryDelta

from agent_platform.memory.adapters.hot_memory_file_adapter import HotMemoryFileAdapter
from agent_platform.memory.adapters.hot_memory_compressor_adapter import (
    TruncatingHotMemoryCompressorAdapter,
)
from agent_platform.memory.adapters.memory_port_adapter import MemoryPortAdapter
from agent_platform.memory.adapters.summarizer_adapter import TruncatingSummarizerAdapter
from agent_platform.storage.adapters.sqlite.relational_adapter import (
    AsyncSQLiteRelationalAdapter,
)

from app.agents.memory.memory_graph_state import (
    append_pending_memory_delta,
    flush_graph_pending_deltas_on_finalize,
    pending_memory_delta_from_ctx,
    rehydrate_working_memory_to_ctx,
    sync_pending_deltas_from_hot,
)
from core.composition.tool_dispatch import invoke_tool


@pytest.fixture
def memory_stack(tmp_path):
    store = str(tmp_path / "memory")
    db = AsyncSQLiteRelationalAdapter(
        db_path=str(tmp_path / "archive.db"), pool_size=2
    )
    hot = HotMemoryFileAdapter(store_dir=store)
    mem = MemoryPortAdapter(
        store_dir=store,
        archive_db=db,
        hot_memory=hot,
        summarizer=TruncatingSummarizerAdapter(max_chars=2000),
        compressor=TruncatingHotMemoryCompressorAdapter(),
    )
    ctx = RunContext(
        request=RequestContext(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            trace_id="tr",
            channel="test",
        ),
        memory=mem,
    )
    ctx.extra = {}
    return ctx, mem, db


@pytest.mark.asyncio
async def test_pending_delta_finalize_loop(memory_stack):
    ctx, mem, db = memory_stack
    try:
        delta = MemoryDelta(key="语言", value="中文", source="user")
        await mem.update_prompt_memory(ctx.request, delta, require_hitl=True)
        append_pending_memory_delta(ctx, delta, require_hitl=True)
        assert len(pending_memory_delta_from_ctx(ctx)) == 1
        await flush_graph_pending_deltas_on_finalize(ctx)
        summary = await mem.finalize_session(ctx.request)
        assert summary.get("pending_applied", 0) >= 1
        snap = mem.compose_prompt_snapshot(ctx.request)
        assert "中文" in snap.memory_text
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_rehydrate_from_graph_state(memory_stack):
    ctx, mem, db = memory_stack
    try:
        state = {
            "retrieval_intent": "knowledge",
            "pending_memory_delta": [
                {"key": "称呼", "value": "小王", "source": "user", "require_hitl": True}
            ],
            "pending_remember": "称呼=小王",
            "memory_path": "path_b",
            "l0_applied": False,
        }
        rehydrate_working_memory_to_ctx(ctx, state)
        assert ctx.extra["retrieval_intent"] == "knowledge"
        assert len(pending_memory_delta_from_ctx(ctx)) == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sync_pending_from_hot_after_restart(memory_stack):
    ctx, mem, db = memory_stack
    try:
        await mem.update_prompt_memory(
            ctx.request,
            MemoryDelta(key="称呼", value="小王", source="user"),
            require_hitl=True,
        )
        ctx.extra = {}
        n = sync_pending_deltas_from_hot(ctx)
        assert n == 1
        assert pending_memory_delta_from_ctx(ctx)[0]["key"] == "称呼"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_tool_dispatch_records_delta(memory_stack):
    ctx, mem, db = memory_stack
    try:
        await invoke_tool(
            ctx,
            "remember_user_fact",
            {"key": "语言", "value": "英文", "require_hitl": True},
        )
        deltas = pending_memory_delta_from_ctx(ctx)
        assert any(d["key"] == "语言" for d in deltas)
    finally:
        await db.close()
