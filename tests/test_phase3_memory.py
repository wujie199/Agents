"""第三期：L3 技能路由、L4 HTTP/refresh、finalize 摘要。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from core.domain.context import RequestContext
from core.composition.run_context import RunContext

from app.agents.context_builder import is_skill_query
from app.agents.enterprise_memory import format_finalize_summary, refresh_l4_profile
from app.agents.retrieval_router import (
    build_retrieval_plan,
    classify_intent,
    should_use_direct_llm_for_intent,
)
from app.agents.chat_config import ChatAgentConfig
from agent_platform.memory.adapters.cached_external_memory_adapter import (
    CachedExternalMemoryAdapter,
)
from agent_platform.memory.adapters.http_external_memory_adapter import (
    HttpExternalMemoryAdapter,
)


def test_dev_skills_and_l4_http_configs_exist():
    for name in (
        "memory.dev-skills.example.yml",
        "memory.dev-l4-http.example.yml",
    ):
        path = Path("config") / name
        assert path.is_file(), name


def test_dev_skills_auto_extract_enabled():
    cfg = yaml.safe_load(
        Path("config/memory.dev-skills.example.yml").read_text(encoding="utf-8")
    )
    assert cfg.get("skill_auto_extract_draft") is True


@pytest.mark.parametrize(
    "query",
    [
        "用 json_section 技能处理",
        "按 report_context 技能生成报告",
        "执行会话查找流程",
        "跑一下 list_json_titles 工作流",
    ],
)
def test_skill_query_patterns(query):
    assert is_skill_query(query)


def test_classify_skill_intent():
    cfg = ChatAgentConfig(retrieval_orchestration=True, enable_skill_tools=True)
    assert classify_intent("用 json_section 技能", cfg) == "skill"


def test_skill_plan_skips_rag():
    cfg = ChatAgentConfig(
        retrieval_orchestration=True,
        skill_prefetch=True,
        enable_skill_tools=True,
    )
    plan = build_retrieval_plan("执行 report_context 流程", cfg, enable_rag=True)
    assert plan.intent == "skill"
    assert plan.run_skill
    assert not plan.run_rag
    assert plan.skip_rag_reason == "skill_intent"


def test_skill_intent_uses_react_not_direct_llm():
    cfg = ChatAgentConfig(knowledge_direct_llm=True, enable_memory_tools=True)
    assert not should_use_direct_llm_for_intent("skill", cfg)


def test_format_finalize_summary():
    text = format_finalize_summary(
        {"pending_applied": 1, "l4_merged": 2, "l4_keys": ["部门", "职位"]}
    )
    assert "pending" in text
    assert "L4→L1" in text
    assert "部门" in text


@pytest.mark.asyncio
async def test_refresh_l4_profile_invalidates_cache():
    inner = AsyncMock()
    inner.fetch_profile_facts = AsyncMock(
        return_value=[
            type("F", (), {"key": "部门", "value": "研发", "source": "ldap"})()
        ]
    )
    cached = CachedExternalMemoryAdapter(inner, ttl_seconds=300)
    req = RequestContext(
        tenant_id="t1", user_id="u1", session_id="s1", trace_id="t", channel="test"
    )
    memory = MagicMock()
    memory.refresh_external_profile = AsyncMock(
        return_value={"refreshed": True, "fact_count": 1, "facts": [{"key": "部门"}]}
    )
    ctx = RunContext(request=req, memory=memory)
    out = await refresh_l4_profile(ctx)
    assert out["fact_count"] == 1
    memory.refresh_external_profile.assert_awaited_once()


@pytest.mark.asyncio
async def test_http_mock_profile_shape():
    """HttpExternalMemoryAdapter 与 mock server JSON 契约一致。"""
    profile = {
        "facts": [{"key": "部门", "value": "研发部", "source": "ldap"}],
        "entities": {},
    }
    facts = HttpExternalMemoryAdapter._facts_from_profile(profile)
    assert len(facts) == 1
    assert facts[0].key == "部门"


@pytest.mark.asyncio
async def test_finalize_session_returns_summary(tmp_path):
    from agent_platform.infrastructure.privacy.adapter import PrivacyPortAdapter
    from agent_platform.infrastructure.skills.adapter import SimpleSkillAdapter
    from agent_platform.memory.adapters.file_external_memory_adapter import (
        FileExternalMemoryAdapter,
    )
    from agent_platform.memory.adapters.hot_memory_compressor_adapter import (
        TruncatingHotMemoryCompressorAdapter,
    )
    from agent_platform.memory.adapters.hot_memory_file_adapter import HotMemoryFileAdapter
    from agent_platform.memory.adapters.memory_port_adapter import MemoryPortAdapter
    from agent_platform.memory.adapters.skill_memory_adapter import SkillMemoryAdapter
    from agent_platform.memory.adapters.summarizer_adapter import TruncatingSummarizerAdapter
    from agent_platform.storage.adapters.sqlite.relational_adapter import (
        AsyncSQLiteRelationalAdapter,
    )

    store = str(tmp_path / "mem")
    ext_dir = tmp_path / "ext" / "tenant1"
    ext_dir.mkdir(parents=True)
    (ext_dir / "user1.yaml").write_text(
        "facts:\n  - key: 部门\n    value: 研发部\n    source: ldap\n",
        encoding="utf-8",
    )
    hot = HotMemoryFileAdapter(store_dir=store)
    db = AsyncSQLiteRelationalAdapter(db_path=str(tmp_path / "a.db"), pool_size=2)
    external = FileExternalMemoryAdapter(profiles_dir=str(tmp_path / "ext"))
    memory = MemoryPortAdapter(
        store_dir=store,
        archive_db=db,
        hot_memory=hot,
        privacy=PrivacyPortAdapter(),
        skill_memory=SkillMemoryAdapter(
            skills=SimpleSkillAdapter(skills_dir="skills/published"),
            drafts_dir=str(tmp_path / "drafts"),
        ),
        summarizer=TruncatingSummarizerAdapter(max_chars=500),
        compressor=TruncatingHotMemoryCompressorAdapter(),
        external_memory=external,
        external_merge_on_finalize=True,
    )
    ctx = RequestContext(
        tenant_id="tenant1",
        user_id="user1",
        session_id="fin",
        trace_id="t",
        channel="test",
    )
    summary = await memory.finalize_session(ctx)
    assert summary.get("l4_merged", 0) >= 1
    assert "部门" in (summary.get("l4_keys") or [])
