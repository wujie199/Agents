import pytest
from datetime import datetime
from pathlib import Path

from core.domain.context import RequestContext
from core.ports.memory import MemoryDelta, SkillOutcome, ToolCallRecord, TurnRecord
from core.composition.run_context import RunContext
from agent_platform.tools.adapters.tool_port_adapter import ToolPortAdapter
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
from agent_platform.storage.adapters.memory.async_cache_adapter import (
    AsyncMemoryCacheAdapter,
)
from agent_platform.storage.adapters.sqlite.relational_adapter import (
    AsyncSQLiteRelationalAdapter,
)
from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter
from agent_platform.memory.adapters.session_message_vector_index import (
    SessionMessageVectorIndex,
)
from document.rag.adapters.embedding.mock import MockEmbeddingModel


def _ctx(session_id: str = "sess1", user_id: str = "user1") -> RequestContext:
    return RequestContext(
        tenant_id="tenant1",
        user_id=user_id,
        session_id=session_id,
        trace_id="trace1",
        channel="test",
    )


def _skill_run_context(ctx: RequestContext, memory) -> RunContext:
    tools = ToolPortAdapter(config_path="config/tools.yml")
    return RunContext(request=ctx, memory=memory, tools=tools)


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
    profile_file = profile_dir / "user1.yaml"
    profile_file.write_text(
        "facts:\n  - key: 部门\n    value: 研发部\n    source: ldap\n",
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
        cache=AsyncMemoryCacheAdapter(prefix="sess"),
    )


@pytest.fixture
def memory_with_vector(tmp_store, tmp_db, external_profiles, tmp_path):
    hot = HotMemoryFileAdapter(store_dir=tmp_store)
    skills = SimpleSkillAdapter(skills_dir="skills/published")
    skill_memory = SkillMemoryAdapter(
        skills=skills,
        drafts_dir=str(Path(tmp_store) / "drafts"),
        meta_dir=str(Path(tmp_store) / "meta"),
    )
    external = FileExternalMemoryAdapter(profiles_dir=external_profiles)
    vector_port = ChromaVectorAdapter(persist_directory=str(tmp_path / "chroma"))
    vector_index = SessionMessageVectorIndex(
        vector_port, MockEmbeddingModel(dimension=64), index_version="mock:64"
    )
    return MemoryPortAdapter(
        store_dir=tmp_store,
        archive_db=tmp_db,
        hot_memory=hot,
        privacy=PrivacyPortAdapter(),
        skill_memory=skill_memory,
        summarizer=TruncatingSummarizerAdapter(max_chars=500),
        compressor=TruncatingHotMemoryCompressorAdapter(),
        external_memory=external,
        cache=AsyncMemoryCacheAdapter(prefix="sess"),
        session_vector_index=vector_index,
        session_hybrid_search=True,
    )


class TestL1HotMemory:
    def test_compose_snapshot_empty(self, memory):
        snap = memory.compose_prompt_snapshot(_ctx())
        assert "# SYSTEM MEMORY" in snap.memory_text
        assert snap.frozen is True

    @pytest.mark.asyncio
    async def test_apply_delta_changes_hash(self, memory):
        ctx = _ctx()
        before = memory.compose_prompt_snapshot(ctx).hash
        await memory.apply_memory_delta(
            ctx, MemoryDelta(key="称呼", value="小明", source="user")
        )
        after = memory.compose_prompt_snapshot(ctx).hash
        assert before != after

    @pytest.mark.asyncio
    async def test_finalize_merges_pending_and_l4(self, memory):
        ctx = _ctx("finalize")
        await memory.update_prompt_memory(
            ctx,
            MemoryDelta(key="称呼", value="待定", source="user"),
            require_hitl=True,
        )
        await memory.finalize_session(ctx)
        snap = memory.compose_prompt_snapshot(ctx)
        assert "称呼: 待定" in snap.memory_text
        assert "部门: 研发部" in snap.memory_text

    @pytest.mark.asyncio
    async def test_confirm_pending_deltas(self, memory):
        ctx = _ctx()
        await memory.update_prompt_memory(
            ctx,
            MemoryDelta(key="k", value="v", source="user"),
            require_hitl=True,
        )
        count = await memory.confirm_pending_deltas(ctx)
        assert count == 1
        assert "k: v" in memory.compose_prompt_snapshot(ctx).memory_text


class TestL2Archive:
    @pytest.mark.asyncio
    async def test_session_lifecycle_with_finalize(self, memory, tmp_db):
        ctx = _ctx("lifecycle")
        await memory.ensure_session(ctx)
        await memory.persist_turn(
            ctx,
            TurnRecord(role="user", content="你好", ts=datetime.now().isoformat()),
        )
        await memory.end_session(ctx, finalize=True)
        ended = await tmp_db.select_one(
            "sessions", ["status"], {"session_id": "lifecycle"}
        )
        assert ended["status"] == "closed"

    @pytest.mark.asyncio
    async def test_session_search_cached(self, memory):
        ctx = _ctx("search_sess")
        await memory.persist_turn(
            ctx,
            TurnRecord(
                role="user",
                content="项目代号 Phoenix",
                ts=datetime.now().isoformat(),
            ),
        )
        r1 = await memory.session_search("Phoenix", ctx, limit=3)
        r2 = await memory.session_search("Phoenix", ctx, limit=3)
        assert "Phoenix" in r1
        assert r1 == r2

    @pytest.mark.asyncio
    async def test_persist_turn_idempotent(self, memory, tmp_db):
        ctx = _ctx("idem")
        turn = TurnRecord(
            role="user", content="repeat me", ts="2025-01-01T00:00:00"
        )
        await memory.persist_turn(ctx, turn)
        await memory.persist_turn(ctx, turn)
        rows = await tmp_db.select_many(
            "messages",
            ["message_id", "content"],
            where={"session_id": "idem"},
        )
        assert len(rows) == 1
        assert rows[0]["content"] == "repeat me"

    @pytest.mark.asyncio
    async def test_session_search_chinese_reorder(self, memory):
        ctx = _ctx("cjk_search")
        await memory.persist_turn(
            ctx,
            TurnRecord(
                role="user",
                content="如何选择扫地机器人",
                ts=datetime.now().isoformat(),
            ),
        )
        result = await memory.session_search(
            "扫地机器人如何选择", ctx, limit=3, scope="session"
        )
        assert result
        assert "扫地" in result or "机器人" in result

    @pytest.mark.asyncio
    async def test_session_search_user_scope(self, memory):
        ctx_a = _ctx("sess_a")
        ctx_b = _ctx("sess_b")
        ts = datetime.now().isoformat()
        await memory.persist_turn(
            ctx_a,
            TurnRecord(role="user", content="Alpha 项目讨论", ts=ts),
        )
        await memory.persist_turn(
            ctx_b,
            TurnRecord(role="user", content="Beta Alpha 备份", ts=ts),
        )
        scoped = await memory.session_search("Alpha", ctx_a, limit=3, scope="session")
        cross = await memory.session_search("Alpha", ctx_a, limit=5, scope="user")
        assert "Alpha" in scoped
        assert "Alpha" in cross
        assert "sess_b" in cross or "Beta" in cross

    @pytest.mark.asyncio
    async def test_cache_invalidated_after_persist(self, memory):
        ctx = _ctx("cache_inv")
        await memory.persist_turn(
            ctx,
            TurnRecord(role="user", content="cached keyword", ts=datetime.now().isoformat()),
        )
        first = await memory.session_search("cached", ctx, limit=3)
        assert "cached" in first.lower()
        await memory.persist_turn(
            ctx,
            TurnRecord(
                role="user",
                content="cached keyword updated secret",
                ts=datetime.now().isoformat(),
            ),
        )
        second = await memory.session_search("secret", ctx, limit=3)
        assert "secret" in second.lower()

    @pytest.mark.asyncio
    async def test_redacted_excluded_from_search(self, memory, tmp_db):
        ctx = _ctx("redact_search")
        await memory.persist_turn(
            ctx,
            TurnRecord(role="user", content="findme token", ts=datetime.now().isoformat()),
        )
        await memory.purge_user_data("tenant1", "user1")
        result = await memory.session_search("findme", ctx, limit=3)
        assert result == ""

    @pytest.mark.asyncio
    async def test_hybrid_vector_search(self, memory_with_vector):
        ctx = _ctx("hybrid_sess")
        content = "semantic vector retrieval about lunar base"
        await memory_with_vector.persist_turn(
            ctx,
            TurnRecord(role="user", content=content, ts=datetime.now().isoformat()),
        )
        result = await memory_with_vector.session_search(
            content, ctx, limit=3, scope="session"
        )
        assert "lunar" in result.lower()

    @pytest.mark.asyncio
    async def test_reindex_session_vectors(self, memory_with_vector, tmp_db):
        ctx = _ctx("reindex_sess")
        await memory_with_vector.persist_turn(
            ctx,
            TurnRecord(
                role="user",
                content="historical message for vector reindex",
                ts=datetime.now().isoformat(),
            ),
        )
        result = await memory_with_vector.reindex_session_vectors(
            tenant_id="tenant1",
            user_id="user1",
            session_id="reindex_sess",
            batch_size=50,
        )
        assert result["indexed"] >= 1
        assert result["errors"] == 0
        search = await memory_with_vector.session_search(
            "historical vector", ctx, limit=3
        )
        assert "historical" in search.lower()

    @pytest.mark.asyncio
    async def test_purge_expired(self, memory, tmp_db):
        ctx = _ctx("old")
        await memory.ensure_session(ctx)
        await tmp_db.update(
            "sessions",
            {"started_at": "2000-01-01T00:00:00"},
            {"session_id": "old"},
        )
        count = await memory.purge_expired_sessions(retention_days=90)
        assert count >= 0


class TestL3SkillSearch:
    @pytest.mark.asyncio
    async def test_skill_search_with_metadata(self, memory):
        hits = await memory.skill_search("报告", _ctx(), limit=3)
        assert any(h.skill_id == "example" for h in hits)
        assert hits[0].success_rate == 1.0

    @pytest.mark.asyncio
    async def test_extract_draft(self, memory, tmp_store):
        skill_memory = memory._skill_memory
        draft_id = skill_memory.extract_draft(
            "tenant1",
            "测试技能",
            ["测试"],
            [{"action": "step1", "tool": None}],
        )
        assert draft_id
        draft_path = Path(tmp_store) / "drafts" / "tenant1" / draft_id / "skill.yaml"
        assert draft_path.exists()

    @pytest.mark.asyncio
    async def test_list_and_get_skill(self, memory):
        ctx = _ctx()
        rows = await memory.list_skills(ctx)
        assert any(r["skill_id"] == "example" for r in rows)
        detail = await memory.get_skill("example", ctx)
        assert detail is not None
        assert detail["title"]
        assert detail["steps"]

    @pytest.mark.asyncio
    async def test_run_skill_with_echo(self, memory):
        ctx = _ctx()
        run_ctx = _skill_run_context(ctx, memory)
        result = await memory.run_skill(
            "example",
            {"message": "hello skill"},
            ctx,
            run_ctx,
        )
        assert result.success
        assert result.outputs["echo"]["echo"] == "hello skill"

    @pytest.mark.asyncio
    async def test_record_outcome_writes_meta_not_published_yaml(
        self, memory, tmp_store
    ):
        ctx = _ctx()
        await memory.record_skill_outcome(
            ctx,
            SkillOutcome(skill_id="example", success=False, error="test failure"),
        )
        meta_path = Path(tmp_store) / "meta" / "tenant1" / "example.json"
        assert meta_path.exists()
        published = Path("skills/published/example/skill.yaml").read_text(
            encoding="utf-8"
        )
        assert "test failure" not in published

    @pytest.mark.asyncio
    async def test_publish_skill(self, memory, tmp_store, tmp_path):
        ctx = _ctx()
        draft_id = await memory.extract_skill_draft(
            ctx,
            "发布测试",
            ["发布"],
            [
                {
                    "action": "echo",
                    "tool": "skill_echo",
                    "args_template": {"message": "$inputs.msg"},
                }
            ],
            skill_id="publish_test",
        )
        assert draft_id == "publish_test"
        pub_root = tmp_path / "published"
        memory._skill_memory._published_dir = pub_root
        result = await memory.publish_skill(ctx, draft_id)
        assert result["success"]
        assert (pub_root / "publish_test" / "skill.yaml").exists()
        drafts = await memory.list_skill_drafts(ctx)
        assert not any(d["skill_id"] == draft_id for d in drafts)


class TestL4External:
    @pytest.mark.asyncio
    async def test_resolve_entity(self, memory, external_profiles):
        entities_dir = Path(external_profiles) / "tenant1"
        entities_dir.mkdir(parents=True, exist_ok=True)
        (entities_dir / "user1.yaml").write_text(
            "entities:\n  张三:\n    canonical_id: u001\n    display_name: 张三\n",
            encoding="utf-8",
        )
        entity = await memory.resolve_entity("张三", _ctx())
        assert entity is not None
        assert entity.canonical_id == "u001"

    @pytest.mark.asyncio
    async def test_purge_user_data(self, memory, tmp_db, external_profiles):
        ctx = _ctx("purge_me")
        await memory.persist_turn(
            ctx,
            TurnRecord(role="user", content="secret", ts=datetime.now().isoformat()),
        )
        profile_path = Path(external_profiles) / "tenant1" / "user1.yaml"
        assert profile_path.exists()
        result = await memory.purge_user_data("tenant1", "user1")
        row = await tmp_db.select_one(
            "messages", ["content"], {"session_id": "purge_me"}
        )
        assert row["content"] == "[redacted]"
        assert result.get("external_profile_deleted") is True
        assert not profile_path.exists()

    @pytest.mark.asyncio
    async def test_session_search_detail_structure(self, memory):
        ctx = _ctx("detail_sess")
        await memory.persist_turn(
            ctx,
            TurnRecord(
                role="user",
                content="Detail search keyword alpha",
                ts=datetime.now().isoformat(),
            ),
        )
        detail = await memory.session_search_detail(
            "Detail alpha", ctx, limit=3, scope="session"
        )
        assert detail.summary
        assert detail.fragments
        assert detail.fragments[0].message_id
        payload = detail.to_dict()
        assert "fragments" in payload
        assert payload["sources"]

    @pytest.mark.asyncio
    async def test_tool_call_search(self, memory, tmp_db):
        ctx = _ctx("tool_sess")
        await memory.ensure_session(ctx)
        await tmp_db.insert_tool_call(
            {
                "call_id": "call1",
                "session_id": ctx.session_id,
                "tool_name": "weather_api",
                "args_hash": "abc",
                "result_summary": "Beijing sunny 25C",
                "status": "ok",
                "latency_ms": 10,
                "ts": datetime.now().isoformat(),
            }
        )
        detail = await memory.session_search_detail(
            "weather_api Beijing", ctx, limit=3, scope="session"
        )
        assert detail.fragments
        assert any(f.role == "tool" for f in detail.fragments)

    @pytest.mark.asyncio
    async def test_tool_call_vector_index(self, memory_with_vector):
        ctx = _ctx("tool_vec_sess")
        await memory_with_vector.ensure_session(ctx)
        await memory_with_vector.persist_tool_call(
            ctx,
            ToolCallRecord(
                tool_name="weather_api",
                result_summary="Shanghai rainy 18C",
                status="ok",
                latency_ms=12,
            ),
        )
        detail = await memory_with_vector.session_search_detail(
            "rainy Shanghai weather", ctx, limit=3, scope="session"
        )
        assert detail.fragments
        assert any("Shanghai" in f.content or "rainy" in f.content.lower() for f in detail.fragments)

    @pytest.mark.asyncio
    async def test_vector_index_version_auto_reindex(self, memory_with_vector):
        ctx = _ctx("vec_version_sess")
        await memory_with_vector.persist_turn(
            ctx,
            TurnRecord(
                role="user",
                content="version reindex seed content",
                ts=datetime.now().isoformat(),
            ),
        )

        memory_with_vector._vector_index._vector.set_index_version(
            memory_with_vector._vector_index._collection, "stale:32"
        )
        memory_with_vector._vector_version_checked = False

        await memory_with_vector.session_search_detail(
            "version reindex seed", ctx, limit=3, scope="session"
        )

        stored = memory_with_vector._vector_index.get_stored_version()
        assert stored == memory_with_vector._vector_index.index_version
