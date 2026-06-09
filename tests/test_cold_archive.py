"""L2 冷归档：对象存储 + 索引表 + 在线库删除。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

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
        self._blobs: dict[str, bytes] = {}

    def upload(self, key: str, data: bytes, content_type=None) -> str:
        self._blobs[key] = data
        return key

    def download(self, key: str):
        return self._blobs.get(key)

    def delete(self, key: str) -> None:
        self._blobs.pop(key, None)


def _ctx(session_id: str = "sess-cold") -> RequestContext:
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
        cache=AsyncMemoryCacheAdapter(prefix="cold"),
        retention_days=90,
        object_store=object_store,
        enable_cold_archive=True,
        cold_archive_prefix="l2/cold/test",
        cold_archive_compress=True,
    ), object_store, db


@pytest.mark.asyncio
async def test_archive_session_moves_to_cold_store(cold_memory):
    memory, store, db = cold_memory
    ctx = _ctx("sess-archive-1")
    await memory.ensure_session(ctx)
    await memory.persist_turn(
        ctx,
        TurnRecord(
            role="user",
            content="hello cold archive",
            ts=datetime.now().isoformat(),
            trace_id="t1",
        ),
    )

    result = await memory.archive_session(ctx.session_id)
    assert result["skipped"] is False
    assert result["message_count"] == 1
    assert result["object_key"] in store._blobs

    online = await db.select_one("sessions", ["session_id"], {"session_id": ctx.session_id})
    assert online is None

    index = await db.get_cold_archive(ctx.session_id)
    assert index is not None
    assert index["message_count"] == 1


@pytest.mark.asyncio
async def test_archive_session_idempotent(cold_memory):
    memory, store, _db = cold_memory
    ctx = _ctx("sess-idempotent")
    await memory.ensure_session(ctx)
    await memory.persist_turn(
        ctx,
        TurnRecord(
            role="user",
            content="once",
            ts=datetime.now().isoformat(),
            trace_id="t1",
        ),
    )
    first = await memory.archive_session(ctx.session_id)
    second = await memory.archive_session(ctx.session_id)
    assert first["skipped"] is False
    assert second["skipped"] is True
    assert len(store._blobs) == 1


@pytest.mark.asyncio
async def test_fetch_cold_session(cold_memory):
    memory, _, _db = cold_memory
    ctx = _ctx("sess-fetch")
    await memory.ensure_session(ctx)
    await memory.persist_turn(
        ctx,
        TurnRecord(
            role="assistant",
            content="stored in cold",
            ts=datetime.now().isoformat(),
            trace_id="t1",
        ),
    )
    await memory.archive_session(ctx.session_id)

    payload = await memory.fetch_cold_session(ctx.session_id)
    assert payload is not None
    assert payload["session"]["session_id"] == ctx.session_id
    assert payload["messages"][0]["content"] == "stored in cold"
    assert "cold_index" in payload


@pytest.mark.asyncio
async def test_list_cold_archives(cold_memory):
    memory, _, _db = cold_memory
    for sid in ("sess-a", "sess-b"):
        ctx = _ctx(sid)
        await memory.ensure_session(ctx)
        await memory.persist_turn(
            ctx,
            TurnRecord(
                role="user",
                content=f"msg-{sid}",
                ts=datetime.now().isoformat(),
                trace_id="t1",
            ),
        )
        await memory.archive_session(sid)

    rows = await memory.list_cold_archives("tenant1", "user1", limit=10)
    assert len(rows) == 2
    session_ids = {r["session_id"] for r in rows}
    assert session_ids == {"sess-a", "sess-b"}


@pytest.mark.asyncio
async def test_archive_expired_sessions(cold_memory):
    memory, store, db = cold_memory
    ctx = _ctx("sess-expired")
    await memory.ensure_session(ctx)
    await memory.persist_turn(
        ctx,
        TurnRecord(
            role="user",
            content="old session",
            ts=datetime.now().isoformat(),
            trace_id="t1",
        ),
    )
    old_start = (datetime.now() - timedelta(days=120)).isoformat()
    async with db._get_connection() as conn:
        cursor = await conn.cursor()
        await cursor.execute(
            """
            UPDATE sessions
            SET started_at = :started_at,
                ended_at = :started_at,
                status = 'closed'
            WHERE session_id = :session_id
            """,
            {"started_at": old_start, "session_id": ctx.session_id},
        )
        await conn.commit()

    result = await memory.archive_expired_sessions(retention_days=90)
    assert result["candidates"] >= 1
    assert result["archived"] >= 1
    assert ctx.session_id not in store._blobs or result["archived"] >= 1
    assert await db.get_cold_archive(ctx.session_id) is not None


@pytest.mark.asyncio
async def test_purge_user_deletes_cold(cold_memory):
    memory, store, db = cold_memory
    ctx = _ctx("sess-purge")
    await memory.ensure_session(ctx)
    await memory.persist_turn(
        ctx,
        TurnRecord(
            role="user",
            content="purge me",
            ts=datetime.now().isoformat(),
            trace_id="t1",
        ),
    )
    archived = await memory.archive_session(ctx.session_id)
    key = archived["object_key"]
    assert key in store._blobs

    await memory.purge_user_data("tenant1", "user1")
    assert key not in store._blobs
    assert await db.get_cold_archive(ctx.session_id) is None


@pytest.mark.asyncio
async def test_active_session_not_archived_by_expired_job(cold_memory):
    memory, store, db = cold_memory
    ctx = _ctx("sess-still-active")
    await memory.ensure_session(ctx)
    await memory.persist_turn(
        ctx,
        TurnRecord(
            role="user",
            content="still running",
            ts=datetime.now().isoformat(),
            trace_id="t1",
        ),
    )
    old_start = (datetime.now() - timedelta(days=120)).isoformat()
    async with db._get_connection() as conn:
        cursor = await conn.cursor()
        await cursor.execute(
            "UPDATE sessions SET started_at = :started_at WHERE session_id = :session_id",
            {"started_at": old_start, "session_id": ctx.session_id},
        )
        await conn.commit()

    result = await memory.archive_expired_sessions(retention_days=90)
    assert result["archived"] == 0
    online = await db.select_one("sessions", ["session_id"], {"session_id": ctx.session_id})
    assert online is not None
    assert await db.get_cold_archive(ctx.session_id) is None
