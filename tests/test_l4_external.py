"""L4 External Provider A 层测试。"""

import pytest
from datetime import datetime
from pathlib import Path

from core.domain.context import RequestContext
from core.ports.memory import MemoryDelta, TurnRecord
from agent_platform.memory.adapters.file_external_memory_adapter import (
    FileExternalMemoryAdapter,
)
from agent_platform.memory.adapters.hot_memory_file_adapter import HotMemoryFileAdapter
from agent_platform.memory.adapters.memory_port_adapter import MemoryPortAdapter
from agent_platform.memory.adapters.skill_memory_adapter import SkillMemoryAdapter
from agent_platform.infrastructure.privacy.adapter import PrivacyPortAdapter
from agent_platform.infrastructure.skills.adapter import SimpleSkillAdapter
from agent_platform.memory.adapters.hot_memory_compressor_adapter import (
    TruncatingHotMemoryCompressorAdapter,
)
from agent_platform.memory.adapters.summarizer_adapter import TruncatingSummarizerAdapter
from agent_platform.storage.adapters.sqlite.relational_adapter import (
    AsyncSQLiteRelationalAdapter,
)


def _ctx(session_id: str = "sess1", user_id: str = "user1") -> RequestContext:
    return RequestContext(
        tenant_id="tenant1",
        user_id=user_id,
        session_id=session_id,
        trace_id="trace1",
        channel="test",
    )


@pytest.fixture
def tmp_store(tmp_path):
    return str(tmp_path / "memory")


@pytest.fixture
def tmp_db(tmp_path):
    return AsyncSQLiteRelationalAdapter(
        db_path=str(tmp_path / "archive.db"), pool_size=2
    )


@pytest.fixture
def external_profiles(tmp_path):
    profile_dir = tmp_path / "external_profiles" / "tenant1"
    profile_dir.mkdir(parents=True)
    (profile_dir / "user1.yaml").write_text(
        "entities:\n  张三:\n    canonical_id: u001\n    display_name: 张三\n"
        "facts:\n  - key: 部门\n    value: 研发部\n    source: ldap\n"
        "  - key: 职位\n    value: 工程师\n    source: hr\n",
        encoding="utf-8",
    )
    return str(tmp_path / "external_profiles")


@pytest.fixture
def memory(tmp_store, tmp_db, external_profiles):
    hot = HotMemoryFileAdapter(store_dir=tmp_store)
    skills = SimpleSkillAdapter(skills_dir="skills/published")
    skill_memory = SkillMemoryAdapter(
        skills=skills,
        drafts_dir=str(Path(tmp_store) / "drafts"),
        meta_dir=str(Path(tmp_store) / "meta"),
    )
    external = FileExternalMemoryAdapter(profiles_dir=external_profiles)
    return MemoryPortAdapter(
        store_dir=tmp_store,
        archive_db=tmp_db,
        hot_memory=hot,
        privacy=PrivacyPortAdapter(),
        skill_memory=skill_memory,
        summarizer=TruncatingSummarizerAdapter(max_chars=500),
        compressor=TruncatingHotMemoryCompressorAdapter(),
        external_memory=external,
    )


class TestL4ExternalLayerA:
    @pytest.mark.asyncio
    async def test_resolve_entity(self, memory):
        entity = await memory.resolve_entity("张三", _ctx())
        assert entity is not None
        assert entity.canonical_id == "u001"

    @pytest.mark.asyncio
    async def test_fetch_profile_facts(self, memory):
        facts = await memory.fetch_profile_facts("tenant1", "user1")
        assert len(facts) == 2
        assert facts[0]["source"] == "ldap"

    @pytest.mark.asyncio
    async def test_list_and_get_profile(self, memory):
        users = await memory.list_profile_users("tenant1")
        assert "user1" in users
        profile = await memory.get_profile("tenant1", "user1")
        assert profile.get("entities")

    @pytest.mark.asyncio
    async def test_set_profile_facts(self, memory, external_profiles):
        count = await memory.set_profile_facts(
            "tenant1",
            "user1",
            [{"key": "语言", "value": "中文", "source": "crm"}],
        )
        assert count == 1
        facts = await memory.fetch_profile_facts("tenant1", "user1")
        assert any(f["key"] == "语言" for f in facts)

    @pytest.mark.asyncio
    async def test_finalize_merges_l4_facts(self, memory, tmp_store):
        ctx = _ctx("finalize_sess")
        await memory.apply_memory_delta(
            ctx, MemoryDelta(key="部门", value="旧部门", source="user")
        )
        await memory.finalize_session(ctx)
        user_path = Path(tmp_store) / "tenant1" / "USER_user1.md"
        content = user_path.read_text(encoding="utf-8")
        assert "部门: 研发部" in content
        assert "职位: 工程师" in content
        assert "旧部门" not in content

    @pytest.mark.asyncio
    async def test_finalize_preserves_freeform_text(self, memory, tmp_store):
        user_path = Path(tmp_store) / "tenant1" / "USER_user1.md"
        user_path.parent.mkdir(parents=True, exist_ok=True)
        user_path.write_text(
            "用户偏好摘要：喜欢简洁回复。\n\n部门: 旧部门\n",
            encoding="utf-8",
        )
        ctx = _ctx("freeform_sess")
        await memory.finalize_session(ctx)
        content = user_path.read_text(encoding="utf-8")
        assert "用户偏好摘要：喜欢简洁回复。" in content
        assert "部门: 研发部" in content

    @pytest.mark.asyncio
    async def test_apply_delta_upserts_same_key(self, memory, tmp_store):
        ctx = _ctx("upsert_sess")
        await memory.apply_memory_delta(
            ctx, MemoryDelta(key="称呼", value="小明", source="user")
        )
        await memory.apply_memory_delta(
            ctx, MemoryDelta(key="称呼", value="阿明", source="user")
        )
        content = (
            Path(tmp_store) / "tenant1" / "USER_user1.md"
        ).read_text(encoding="utf-8")
        assert content.count("称呼:") == 1
        assert "称呼: 阿明" in content

    @pytest.mark.asyncio
    async def test_finalize_merge_audit(self, memory, tmp_db):
        ctx = _ctx("audit_sess")
        await memory.finalize_session(ctx)
        rows = await tmp_db.select_many(
            "compliance_audit_log",
            ["resource_type", "action", "resource_id"],
            {"tenant_id": "tenant1", "user_id": "user1"},
        )
        merges = [
            r
            for r in rows
            if r.get("resource_type") == "external_fact"
            and r.get("action") == "merge"
        ]
        assert merges
        assert any(r.get("resource_id") == "部门" for r in merges)

    @pytest.mark.asyncio
    async def test_purge_user_deletes_external_profile(
        self, memory, external_profiles
    ):
        profile_path = Path(external_profiles) / "tenant1" / "user1.yaml"
        assert profile_path.exists()
        result = await memory.purge_user_data("tenant1", "user1")
        assert result.get("external_profile_deleted") is True
        assert not profile_path.exists()

    @pytest.mark.asyncio
    async def test_purge_tenant_l4(self, memory, external_profiles):
        (Path(external_profiles) / "tenant1" / "user2.yaml").write_text(
            "facts: []\n", encoding="utf-8"
        )
        result = await memory.purge_tenant_l4("tenant1")
        assert result["profiles_deleted"] == 2
        assert not (Path(external_profiles) / "tenant1").exists() or not list(
            (Path(external_profiles) / "tenant1").glob("*.yaml")
        )

    @pytest.mark.asyncio
    async def test_import_profile(self, memory, tmp_path):
        src = tmp_path / "import.yaml"
        src.write_text(
            "facts:\n  - key: 项目\n    value: Alpha\n    source: crm\n",
            encoding="utf-8",
        )
        import yaml

        profile = yaml.safe_load(src.read_text(encoding="utf-8"))
        await memory.import_profile("tenant1", "user9", profile)
        facts = await memory.fetch_profile_facts("tenant1", "user9")
        assert facts[0]["value"] == "Alpha"
