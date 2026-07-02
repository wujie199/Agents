"""第二期：冷归档配置、会话列表 enriched、health、turn_decision。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.domain.context import RequestContext
from core.composition.run_context import RunContext

from app.agents.memory.enterprise_memory import list_user_sessions_enriched
from app.agents.memory.memory_metrics import record_turn_decision


from agent_platform.memory.adapters.config_loader import load_memory_config


def test_cold_profile_in_memory_yml(monkeypatch):
    monkeypatch.setenv("MEMORY_PROFILE", "cold")
    cfg = load_memory_config("config/memory.yml")
    assert cfg.get("enable_cold_archive") is True
    assert cfg.get("session_search_cold_fallback") is True
    assert cfg.get("enable_session_vector_index") is False


@pytest.mark.asyncio
async def test_list_user_sessions_enriched_merges_cold():
    req = RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        trace_id="tr",
        channel="test",
    )
    memory = MagicMock()
    memory.list_sessions = AsyncMock(
        return_value=[
            {
                "session_id": "online-1",
                "status": "closed",
                "started_at": "2026-01-02",
            }
        ]
    )
    memory.list_cold_archives = AsyncMock(
        return_value=[
            {
                "session_id": "cold-1",
                "started_at": "2026-01-01",
                "archived_at": "2026-02-01",
                "message_count": 3,
            }
        ]
    )
    ctx = RunContext(request=req, memory=memory)
    rows = await list_user_sessions_enriched(ctx, limit=10)
    ids = {r["session_id"]: r["storage"] for r in rows}
    assert ids["online-1"] == "online_closed"
    assert ids["cold-1"] == "cold"


def test_record_turn_decision_stores_in_extra():
    req = RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        trace_id="tr",
        channel="test",
    )
    ctx = RunContext(request=req, extra={})
    record_turn_decision(
        ctx,
        {"intent": "knowledge", "run_rag": True, "recall_prefetch_hit": False},
    )
    assert ctx.extra["turn_decision"]["intent"] == "knowledge"


def test_memory_health_reports_object_store_when_cold(tmp_path):
    from agent_platform.memory.adapters.hot_memory_file_adapter import (
        HotMemoryFileAdapter,
    )
    from agent_platform.memory.adapters.memory_port_adapter import MemoryPortAdapter
    from agent_platform.memory.adapters.session_cold_archive_service import (
        SessionColdArchiveService,
    )

    class _Store:
        def health(self):
            return {"status": "healthy", "type": "local"}

    hot = HotMemoryFileAdapter(store_dir=str(tmp_path / "mem"))
    cold = SessionColdArchiveService(None, _Store(), prefix="l2/cold")
    memory = MemoryPortAdapter(
        store_dir=str(tmp_path / "mem"),
        hot_memory=hot,
    )
    memory._cold_archive = cold  # type: ignore[attr-defined]
    health = memory.health()
    assert health.get("object_store", {}).get("status") == "healthy"
