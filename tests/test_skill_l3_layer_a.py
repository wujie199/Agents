import json
from pathlib import Path

import pytest

from core.domain.context import RequestContext
from core.composition.run_context import RunContext
from agent_platform.infrastructure.skills.adapter import SimpleSkillAdapter
from agent_platform.memory.adapters.skill_memory_adapter import SkillMemoryAdapter
from agent_platform.memory.adapters.skill_meta_store import SkillMetaStore
from agent_platform.tools.adapters.tool_port_adapter import ToolPortAdapter
from agent_platform.storage.adapters.sqlite.relational_adapter import (
    AsyncSQLiteRelationalAdapter,
)


def _ctx(tenant: str = "tenant1") -> RequestContext:
    return RequestContext(
        tenant_id=tenant,
        user_id="user1",
        session_id="s1",
        trace_id="t1",
        channel="test",
    )


@pytest.mark.asyncio
async def test_meta_tenant_scoped(tmp_path):
    store = SkillMetaStore(meta_dir=str(tmp_path / "meta"))
    store.record_outcome(
        "skill_a", tenant_id="tenant_a", success=True, error=None
    )
    meta = store.load("skill_a", "tenant_a")
    assert meta["usage_count"] == 1
    assert store.load("skill_a", "tenant_b") == {}


@pytest.mark.asyncio
async def test_search_includes_anti_patterns(tmp_path):
    meta_dir = tmp_path / "meta"
    store = SkillMetaStore(str(meta_dir))
    store.save(
        "example",
        {
            "success_rate": 0.8,
            "anti_patterns": ["no source"],
            "usage_count": 3,
            "status": "active",
        },
        tenant_id="tenant1",
    )
    skills = SimpleSkillAdapter(skills_dir="skills/published")
    sm = SkillMemoryAdapter(skills=skills, meta_dir=str(meta_dir))
    hits = sm.search("报告", "tenant1", limit=5)
    demo = next((h for h in hits if h.skill_id == "example"), None)
    assert demo is not None
    assert "Anti-patterns" in demo.summary
    assert demo.usage_count == 3


@pytest.mark.asyncio
async def test_trigger_regex_match(tmp_path):
    skill_dir = tmp_path / "skills" / "rx"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.yaml").write_text(
        """
skill_id: rx_skill
version: "1.0.0"
title: Regex Skill
triggers: []
trigger_regex:
  - "^会话.*检索$"
required_tools: []
max_duration_seconds: 10
acl: [test]
steps: []
""",
        encoding="utf-8",
    )
    adapter = SimpleSkillAdapter(skills_dir=str(tmp_path / "skills"))
    hits = adapter.search("会话内容检索", "tenant1", limit=5)
    assert any(h["skill_id"] == "rx_skill" for h in hits)


@pytest.mark.asyncio
async def test_deprecate_hides_from_search(tmp_path):
    meta_dir = tmp_path / "meta"
    store = SkillMetaStore(str(meta_dir))
    store.set_status("demo", "deprecated", tenant_id="tenant1")
    skills = SimpleSkillAdapter(skills_dir="skills/published")

    class _Skill:
        def search(self, q, t, limit=3):
            return [
                {
                    "skill_id": "demo",
                    "title": "Demo",
                    "triggers": ["demo"],
                    "summary": "x",
                }
            ]

        def get(self, sid):
            return None

        def list_skills(self, t):
            return []

    sm = SkillMemoryAdapter(skills=_Skill(), meta_dir=str(meta_dir))
    assert sm.search("demo", "tenant1") == []


@pytest.mark.asyncio
async def test_skill_run_audit_and_purge(tmp_path):
    db = AsyncSQLiteRelationalAdapter(
        db_path=str(tmp_path / "archive.db"), pool_size=2
    )
    skills = SimpleSkillAdapter(skills_dir="skills/published")
    sm = SkillMemoryAdapter(
        skills=skills,
        meta_dir=str(tmp_path / "meta"),
        archive_db=db,
    )
    tools = ToolPortAdapter(config_path="config/tools.yml")
    ctx = RunContext(request=_ctx(), tools=tools)
    result = await sm.run_and_finalize(
        "example",
        {"message": "audit"},
        ctx,
        _ctx(),
    )
    assert result.success
    rows = await db.select_many(
        "skill_runs",
        ["run_id", "skill_id", "success"],
        where={"tenant_id": "tenant1", "user_id": "user1"},
    )
    assert len(rows) == 1
    purged = await sm.purge_l3_for_user_async("tenant1", "user1")
    assert purged["skill_runs_deleted"] == 1


@pytest.mark.asyncio
async def test_list_skill_runs_and_purge_tenant(tmp_path):
    db = AsyncSQLiteRelationalAdapter(
        db_path=str(tmp_path / "archive.db"), pool_size=2
    )
    skills = SimpleSkillAdapter(skills_dir="skills/published")
    meta_dir = tmp_path / "meta"
    sm = SkillMemoryAdapter(
        skills=skills,
        meta_dir=str(meta_dir),
        drafts_dir=str(tmp_path / "drafts"),
        archive_db=db,
    )
    sm._meta.save("demo", {"usage_count": 1}, tenant_id="tenant1")
    sm.extract_draft("tenant1", "Draft", ["d"], [{"action": "x", "tool": None}])
    tools = ToolPortAdapter(config_path="config/tools.yml")
    from agent_platform.memory.memory_tool_registration import register_memory_tools

    class _Mem:
        async def skill_search(self, *a, **k):
            return []

    mem = _Mem()
    register_memory_tools(tools, mem)
    ctx = RunContext(request=_ctx(), memory=mem, tools=tools)
    await sm.run_and_finalize("example", {"message": "x"}, ctx, _ctx())

    rows = await sm.list_skill_runs("tenant1", user_id="user1")
    assert len(rows) == 1

    purged = await sm.purge_l3_for_tenant_async("tenant1")
    assert purged["meta_files_removed"] >= 1
    assert purged["drafts_removed"] >= 1
    assert purged["skill_runs_deleted"] == 1
    assert await sm.list_skill_runs("tenant1") == []


@pytest.mark.asyncio
async def test_on_session_end_auto_extract(tmp_path):
    db = AsyncSQLiteRelationalAdapter(
        db_path=str(tmp_path / "archive.db"), pool_size=2
    )
    skills = SimpleSkillAdapter(skills_dir="skills/published")
    drafts_dir = tmp_path / "drafts"
    sm = SkillMemoryAdapter(
        skills=skills,
        meta_dir=str(tmp_path / "meta"),
        drafts_dir=str(drafts_dir),
        archive_db=db,
        auto_extract_draft=True,
        auto_extract_min_steps=1,
    )
    tools = ToolPortAdapter(config_path="config/tools.yml")
    ctx = RunContext(request=_ctx(), tools=tools)
    await sm.run_and_finalize("example", {"message": "audit"}, ctx, _ctx())
    end = await sm.on_session_end(_ctx())
    assert end["enabled"] is True
    assert end["runs_scanned"] >= 1
    assert any(d.get("skill_id") == "example" for d in end.get("drafts", []))
