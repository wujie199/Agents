"""L4 Redis 缓存装饰器测试（进程内 cache_port mock）。"""

from __future__ import annotations

import pytest

from core.domain.context import RequestContext
from agent_platform.memory.adapters.cached_external_memory_adapter import (
    CachedExternalMemoryAdapter,
)
from agent_platform.memory.adapters.file_external_memory_adapter import (
    FileExternalMemoryAdapter,
)


class _MemCache:
    def __init__(self):
        self._data: dict = {}

    async def get(self, key: str):
        return self._data.get(key)

    async def set(self, key: str, value, ttl_seconds=None):
        self._data[key] = value

    async def invalidate_pattern(self, pattern: str) -> int:
        prefix = pattern.rstrip("*")
        keys = [k for k in self._data if k.startswith(prefix)]
        for k in keys:
            del self._data[k]
        return len(keys)


@pytest.fixture
def profiles(tmp_path):
    d = tmp_path / "tenant1"
    d.mkdir(parents=True)
    (d / "user1.yaml").write_text(
        "facts:\n  - key: 部门\n    value: 研发\n    source: ldap\n",
        encoding="utf-8",
    )
    return str(tmp_path)


@pytest.mark.asyncio
async def test_l4_cache_hit_with_cache_port(profiles):
    inner = FileExternalMemoryAdapter(profiles_dir=profiles)
    cache = _MemCache()
    adapter = CachedExternalMemoryAdapter(inner, ttl_seconds=60, cache_port=cache)
    ctx = RequestContext(
        tenant_id="tenant1",
        user_id="user1",
        session_id="s1",
        trace_id="t",
        channel="test",
    )
    a = await adapter.fetch_profile_facts("user1", "tenant1")
    b = await adapter.fetch_profile_facts("user1", "tenant1")
    assert len(a) == 1
    assert a == b
    assert adapter.cache_stats["hit"] >= 1
