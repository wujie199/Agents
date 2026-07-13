# -*- coding: utf-8
"""LangGraph Store registry 测试。"""

from __future__ import annotations

import os

import pytest

from agent_platform.memory.adapters.l1_langgraph_store_registry import (
    get_postgres_memory_store,
    teardown_postgres_memory_store,
)
from agent_platform.memory.adapters.hot_memory_langgraph_store_adapter import (
    build_langgraph_memory_store,
)


def test_inmemory_store_singleton_per_call():
    s1 = build_langgraph_memory_store({})
    s2 = build_langgraph_memory_store({})
    assert s1.__class__.__name__ == "InMemoryStore"
    assert s2.__class__.__name__ == "InMemoryStore"


@pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set",
)
def test_postgres_store_setup_teardown():
    url = os.environ["DATABASE_URL"]
    teardown_postgres_memory_store()
    from agent_platform.memory.adapters.l1_langgraph_store_registry import (
        setup_postgres_memory_store,
    )

    store = setup_postgres_memory_store(url)
    assert store is get_postgres_memory_store()
    store.put(("memory", "_test"), "ping", {"content": "pong"})
    assert store.get(("memory", "_test"), "ping") is not None
    teardown_postgres_memory_store()
    assert get_postgres_memory_store() is None
