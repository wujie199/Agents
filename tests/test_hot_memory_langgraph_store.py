# -*- coding: utf-8
"""LangGraph Store L1 后端测试。"""

from __future__ import annotations

from core.domain.context import RequestContext
from core.ports.memory import MemoryDelta

from agent_platform.memory.adapters.hot_memory_langgraph_store_adapter import (
    HotMemoryLangGraphStoreAdapter,
    build_langgraph_memory_store,
)
from core.composition.memory_helpers import build_hot_memory


def test_langgraph_store_compose_and_apply_delta():
    store = build_langgraph_memory_store({})
    hot = HotMemoryLangGraphStoreAdapter(store)
    ctx = RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        trace_id="tr",
        channel="test",
    )
    hot.apply_delta("t1", "u1", MemoryDelta(key="语言", value="中文", source="user"))
    snap = hot.compose_snapshot(ctx)
    assert "中文" in snap.memory_text
    assert snap.hash


def test_build_hot_memory_langgraph_backend():
    hot = build_hot_memory({"l1_store_backend": "langgraph"})
    assert hot.__class__.__name__ == "HotMemoryLangGraphStoreAdapter"
