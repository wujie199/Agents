import pytest
from datetime import datetime

from core.domain.context import RequestContext
from core.ports.memory import TurnRecord
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


class _MemObjectStore:
    def __init__(self):
        self._blobs = {}

    def upload(self, key, data, content_type=None):
        self._blobs[key] = data
        return key

    def download(self, key):
        return self._blobs.get(key)

    def delete(self, key):
        self._blobs.pop(key, None)


def _ctx(session_id: str = "sess-cold-search") -> RequestContext:
    return RequestContext(
        tenant_id="tenant1",
        user_id="user1",
        session_id=session_id,
        trace_id="trace1",
        channel="test",
    )


@pytest.fixture
def cold_memory(tmp_path):
    store_dir = str(tmp_path / "memory")
    db = AsyncSQLiteRelationalAdapter(
        db_path=str(tmp_path / "archive.db"), pool_size=2
    )
    hot = HotMemoryFileAdapter(store_dir=store_dir)
    skills = SimpleSkillAdapter(skills_dir="skills/published")
    skill_memory = SkillMemoryAdapter(
        skills=skills, drafts_dir=str(tmp_path / "drafts")
    )
    external = FileExternalMemoryAdapter(
        profiles_dir=str(tmp_path / "external_profiles")
    )
    object_store = _MemObjectStore()
    return MemoryPortAdapter(
        store_dir=store_dir,
        archive_db=db,
        hot_memory=hot,
        privacy=PrivacyPortAdapter(),
        skill_memory=skill_memory,
        summarizer=TruncatingSummarizerAdapter(max_chars=500),
        compressor=TruncatingHotMemoryCompressorAdapter(),
        external_memory=external,
        cache=AsyncMemoryCacheAdapter(prefix="coldsearch"),
        object_store=object_store,
        enable_cold_archive=True,
        cold_archive_prefix="l2/cold/test",
        session_search_cold_fallback=True,
        session_search_rerank=True,
    ), db


@pytest.mark.asyncio
async def test_session_search_finds_cold_archived_content(cold_memory):
    memory, _db = cold_memory
    ctx = _ctx("sess-only-cold")
    await memory.ensure_session(ctx)
    await memory.persist_turn(
        ctx,
        TurnRecord(
            role="user",
            content="Phoenix cold archive secret keyword",
            ts=datetime.now().isoformat(),
            trace_id="t1",
        ),
    )
    await memory.archive_session(ctx.session_id)

    detail = await memory.session_search_detail(
        "Phoenix cold", ctx, limit=3, scope="session"
    )
    assert detail.fragments
    assert "cold" in detail.sources
    assert "Phoenix" in detail.summary or any(
        "Phoenix" in f.content for f in detail.fragments
    )


@pytest.mark.asyncio
async def test_cold_search_by_session_id_bypasses_scan_limit(cold_memory):
    """指定 session_id 时直接查索引，不受 scan_limit 限制。"""
    memory, _db = cold_memory
    service = memory._cold_archive
    assert service is not None
    service._search_scan_limit = 2

    for i in range(5):
        sid = f"sess-scan-{i}"
        ctx = _ctx(sid)
        await memory.ensure_session(ctx)
        content = "unique elderberry keyword" if i == 0 else f"filler message {i}"
        await memory.persist_turn(
            ctx,
            TurnRecord(
                role="user",
                content=content,
                ts=datetime.now().isoformat(),
                trace_id="t1",
            ),
        )
        await memory.archive_session(sid)

    hits = await service.search_cold_archives(
        "elderberry",
        "tenant1",
        "user1",
        session_id="sess-scan-0",
        limit=5,
    )
    assert hits
    assert any("elderberry" in h.get("content", "") for h in hits)


@pytest.mark.asyncio
async def test_fetch_cold_session_rejects_checksum_mismatch(cold_memory):
    memory, _db = cold_memory
    ctx = _ctx("sess-bad-checksum")
    await memory.ensure_session(ctx)
    await memory.persist_turn(
        ctx,
        TurnRecord(
            role="user",
            content="checksum test payload",
            ts=datetime.now().isoformat(),
            trace_id="t1",
        ),
    )
    await memory.archive_session(ctx.session_id)

    index = await _db.get_cold_archive(ctx.session_id)
    assert index is not None
    await _db.update(
        "cold_archive_sessions",
        {"checksum": "0" * 64},
        {"session_id": ctx.session_id},
    )

    payload = await memory.fetch_cold_session(ctx.session_id)
    assert payload is None


@pytest.mark.asyncio
async def test_cold_search_uses_db_index_without_blob_download(cold_memory):
    memory, db = cold_memory
    ctx = _ctx("sess-db-index")
    await memory.ensure_session(ctx)
    await memory.persist_turn(
        ctx,
        TurnRecord(
            role="user",
            content="database cold index unique term",
            ts=datetime.now().isoformat(),
            trace_id="t1",
        ),
    )
    await memory.archive_session(ctx.session_id)

    rows = await db.select_many(
        "cold_archive_search",
        ["record_id", "content"],
        where={"session_id": ctx.session_id},
    )
    assert len(rows) == 1
    assert "database cold index" in rows[0]["content"]

    service = memory._cold_archive
    assert service is not None

    class _BrokenStore:
        def download(self, key):
            raise RuntimeError("blob download should not be called")

    service._store = _BrokenStore()
    hits = await service.search_cold_archives(
        "database cold index",
        "tenant1",
        "user1",
        session_id=ctx.session_id,
        limit=5,
    )
    assert hits
    assert any("database cold index" in h.get("content", "") for h in hits)
