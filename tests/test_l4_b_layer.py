"""L4 B 层：缓存、工厂、合规 purge、KV 解析。"""

import pytest
from pathlib import Path

from core.domain.context import RequestContext
from core.ports.memory import MemoryDelta
from agent_platform.memory.adapters.cached_external_memory_adapter import (
    CachedExternalMemoryAdapter,
)
from agent_platform.memory.adapters.external_factory import build_external_memory
from agent_platform.memory.adapters.file_external_memory_adapter import (
    FileExternalMemoryAdapter,
)
from agent_platform.memory.adapters.hot_memory_file_adapter import HotMemoryFileAdapter
from agent_platform.memory.adapters.memory_port_adapter import MemoryPortAdapter
from agent_platform.memory.adapters.noop_external_adapter import (
    NoOpExternalMemoryAdapter,
)
from agent_platform.infrastructure.privacy.adapter import PrivacyPortAdapter
from agent_platform.storage.adapters.sqlite.relational_adapter import (
    AsyncSQLiteRelationalAdapter,
)


def _ctx(user_id: str = "user1") -> RequestContext:
    return RequestContext(
        tenant_id="tenant1",
        user_id=user_id,
        session_id="s1",
        trace_id="t",
        channel="test",
    )


class TestExternalFactory:
    def test_build_file_backend(self, tmp_path):
        cfg = {
            "external_profiles_backend": "file",
            "external_profiles_dir": str(tmp_path / "profiles"),
            "external_profile_cache_ttl": 0,
        }
        adapter = build_external_memory(cfg)
        assert isinstance(adapter, FileExternalMemoryAdapter)

    def test_build_noop_backend(self):
        adapter = build_external_memory({"external_profiles_backend": "noop"})
        assert isinstance(adapter, NoOpExternalMemoryAdapter)

    def test_build_with_cache(self, tmp_path):
        cfg = {
            "external_profiles_backend": "file",
            "external_profiles_dir": str(tmp_path / "profiles"),
            "external_profile_cache_ttl": 60,
        }
        adapter = build_external_memory(cfg)
        assert isinstance(adapter, CachedExternalMemoryAdapter)


class TestCachedExternalMemory:
    @pytest.mark.asyncio
    async def test_fetch_facts_cached(self, tmp_path):
        profile_dir = tmp_path / "profiles" / "tenant1"
        profile_dir.mkdir(parents=True)
        (profile_dir / "u1.yaml").write_text(
            "facts:\n  - key: 部门\n    value: A\n    source: ldap\n",
            encoding="utf-8",
        )
        inner = FileExternalMemoryAdapter(profiles_dir=str(tmp_path / "profiles"))
        cached = CachedExternalMemoryAdapter(inner, ttl_seconds=60)
        facts1 = await cached.fetch_profile_facts("u1", "tenant1")
        (profile_dir / "u1.yaml").write_text(
            "facts:\n  - key: 部门\n    value: B\n    source: ldap\n",
            encoding="utf-8",
        )
        facts2 = await cached.fetch_profile_facts("u1", "tenant1")
        assert facts1[0].value == facts2[0].value == "A"

        cached.invalidate_all()
        facts3 = await cached.fetch_profile_facts("u1", "tenant1")
        assert facts3[0].value == "B"


class TestHotMemoryKvAndMemoryUpsert:
    def test_colon_without_space_not_kv(self, tmp_path):
        hot = HotMemoryFileAdapter(store_dir=str(tmp_path / "mem"))
        assert hot._parse_kv_line("http://example.com") is None
        assert hot._parse_kv_line("Note — use Chinese") is None
        assert hot._parse_kv_line("部门: 研发") == ("部门", "研发")

    def test_memory_delta_upsert(self, tmp_path):
        hot = HotMemoryFileAdapter(store_dir=str(tmp_path / "mem"))
        hot.apply_delta(
            "t1", "u1", MemoryDelta(key="规则", value="v1", source="memory")
        )
        hot.apply_delta(
            "t1", "u1", MemoryDelta(key="规则", value="v2", source="memory")
        )
        content = hot.get_raw_memory("t1")
        assert content.count("规则:") == 1
        assert "规则: v2" in content

    def test_strip_user_keys_preserves_freeform(self, tmp_path):
        hot = HotMemoryFileAdapter(store_dir=str(tmp_path / "mem"))
        path = tmp_path / "mem" / "t1" / "USER_u1.md"
        path.parent.mkdir(parents=True)
        path.write_text(
            "摘要段落\n\n部门: 研发\n职位: 工程师\n",
            encoding="utf-8",
        )
        removed = hot.strip_user_keys("t1", "u1", ["部门"])
        content = path.read_text(encoding="utf-8")
        assert removed == 1
        assert "摘要段落" in content
        assert "部门:" not in content
        assert "职位: 工程师" in content


class TestL4CompliancePurge:
    @pytest.fixture
    def setup(self, tmp_path):
        profiles = tmp_path / "profiles"
        store = tmp_path / "store"
        profile_dir = profiles / "tenant1"
        profile_dir.mkdir(parents=True)
        (profile_dir / "user1.yaml").write_text(
            "facts:\n  - key: 部门\n    value: 研发\n    source: ldap\n",
            encoding="utf-8",
        )
        user_path = store / "tenant1" / "USER_user1.md"
        user_path.parent.mkdir(parents=True)
        user_path.write_text("部门: 旧值\n", encoding="utf-8")
        db = AsyncSQLiteRelationalAdapter(
            db_path=str(tmp_path / "db.sqlite"), pool_size=2
        )
        external = FileExternalMemoryAdapter(profiles_dir=str(profiles))
        memory = MemoryPortAdapter(
            store_dir=str(store),
            archive_db=db,
            hot_memory=HotMemoryFileAdapter(store_dir=str(store)),
            privacy=PrivacyPortAdapter(),
            external_memory=external,
            purge_delete_external_audit=True,
            purge_tenant_l4_strip_user_keys=True,
        )
        return memory, db, profiles, user_path

    @pytest.mark.asyncio
    async def test_purge_tenant_strips_user_and_audit(self, setup):
        memory, db, profiles, user_path = setup
        ctx = _ctx()
        await memory.finalize_session(ctx)

        result = await memory.purge_tenant_l4("tenant1")
        assert result["profiles_deleted"] == 1
        assert result["user_keys_stripped"] >= 1
        assert result["external_audit_deleted"] >= 1
        assert "部门:" not in user_path.read_text(encoding="utf-8")
        assert not list((profiles / "tenant1").glob("*.yaml"))

    @pytest.mark.asyncio
    async def test_purge_user_strips_l4_keys_and_audit(self, setup):
        memory, db, _, user_path = setup
        ctx = _ctx()
        await memory.finalize_session(ctx)
        result = await memory.purge_user_data("tenant1", "user1")
        assert result.get("external_profile_deleted") is True
        assert result.get("user_l4_keys_cleared", 0) >= 1
        assert result.get("external_audit_deleted", 0) >= 1
        assert not user_path.exists()

    @pytest.mark.asyncio
    async def test_external_merge_disabled(self, tmp_path):
        profiles = tmp_path / "profiles"
        store = tmp_path / "store"
        (profiles / "tenant1").mkdir(parents=True)
        (profiles / "tenant1" / "user1.yaml").write_text(
            "facts:\n  - key: 部门\n    value: 研发\n    source: ldap\n",
            encoding="utf-8",
        )
        memory = MemoryPortAdapter(
            store_dir=str(store),
            hot_memory=HotMemoryFileAdapter(store_dir=str(store)),
            privacy=PrivacyPortAdapter(),
            external_memory=FileExternalMemoryAdapter(profiles_dir=str(profiles)),
            external_merge_on_finalize=False,
        )
        await memory.finalize_session(_ctx())
        user_path = store / "tenant1" / "USER_user1.md"
        assert not user_path.exists() or "部门:" not in user_path.read_text(
            encoding="utf-8"
        )
