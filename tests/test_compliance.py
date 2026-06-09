"""合规：审计日志、backfill、RAG purge。"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock

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
from agent_platform.memory.adapters.rag_purge_utils import (
    list_rag_document_ids_for_user,
)
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


@pytest.fixture
def compliance_memory(tmp_path):
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
    index_port = AsyncMock()
    index_port.delete_document = AsyncMock(return_value=True)

    memory = MemoryPortAdapter(
        store_dir=store_dir,
        archive_db=db,
        hot_memory=hot,
        privacy=PrivacyPortAdapter(),
        skill_memory=skill_memory,
        summarizer=TruncatingSummarizerAdapter(max_chars=500),
        compressor=TruncatingHotMemoryCompressorAdapter(),
        external_memory=external,
        cache=AsyncMemoryCacheAdapter(prefix="compliance"),
        object_store=object_store,
        enable_cold_archive=True,
        cold_archive_prefix="l2/cold/test",
        index_port=index_port,
    )
    return memory, db, index_port


@pytest.mark.asyncio
async def test_anonymize_writes_audit_log_and_content_hash(compliance_memory):
    memory, db, _ = compliance_memory
    ctx = RequestContext(
        tenant_id="tenant1",
        user_id="user1",
        session_id="audit_sess",
        trace_id="t1",
        channel="test",
    )
    await memory.ensure_session(ctx)
    await memory.persist_turn(
        ctx,
        TurnRecord(
            role="user",
            content="secret compliance payload",
            ts=datetime.now().isoformat(),
        ),
    )

    count = await db.anonymize_user_data("tenant1", "user1")
    assert count == 1

    row = await db.select_one(
        "messages",
        ["content", "content_hash", "redacted"],
        {"session_id": "audit_sess"},
    )
    assert row["content"] == "[redacted]"
    assert row["redacted"] == 1
    assert row["content_hash"]
    assert len(row["content_hash"]) == 64

    audits = await db.select_many(
        "compliance_audit_log",
        ["resource_type", "resource_id", "content_hash", "action"],
        where={"tenant_id": "tenant1", "user_id": "user1"},
        limit=10,
    )
    assert audits
    assert audits[0]["action"] == "anonymize"
    assert audits[0]["content_hash"] == row["content_hash"]


@pytest.mark.asyncio
async def test_purge_user_links_rag_delete(compliance_memory):
    memory, db, index_port = compliance_memory
    await db.insert(
        "documents",
        {
            "doc_id": "doc-u1",
            "tenant_id": "tenant1",
            "title": "user doc",
            "content": "preview",
            "metadata": json.dumps({"user_id": "user1"}),
        },
    )

    ctx = RequestContext(
        tenant_id="tenant1",
        user_id="user1",
        session_id="purge_rag",
        trace_id="t1",
        channel="test",
    )
    await memory.ensure_session(ctx)
    await memory.persist_turn(
        ctx,
        TurnRecord(role="user", content="hi", ts=datetime.now().isoformat()),
    )

    summary = await memory.purge_user_data("tenant1", "user1")
    assert summary["rag_documents_deleted"] == 1
    assert "doc-u1" in summary["rag_doc_ids"]
    index_port.delete_document.assert_awaited_with("doc-u1", "tenant1")


@pytest.mark.asyncio
async def test_list_rag_document_ids_for_user(compliance_memory):
    _, db, _ = compliance_memory
    await db.insert(
        "documents",
        {
            "doc_id": "owned",
            "tenant_id": "tenant1",
            "title": "t",
            "content": "c",
            "metadata": json.dumps({"owner_id": "user1"}),
        },
    )
    await db.insert(
        "documents",
        {
            "doc_id": "other",
            "tenant_id": "tenant1",
            "title": "t2",
            "content": "c2",
            "metadata": json.dumps({"user_id": "user2"}),
        },
    )
    ids = await list_rag_document_ids_for_user(db, "tenant1", "user1")
    assert ids == ["owned"]


@pytest.mark.asyncio
async def test_backfill_cold_search_index(compliance_memory):
    memory, db, _ = compliance_memory
    ctx = RequestContext(
        tenant_id="tenant1",
        user_id="user1",
        session_id="backfill_sess",
        trace_id="t1",
        channel="test",
    )
    await memory.ensure_session(ctx)
    await memory.persist_turn(
        ctx,
        TurnRecord(
            role="user",
            content="legacy cold backfill keyword",
            ts=datetime.now().isoformat(),
        ),
    )
    await memory.archive_session(ctx.session_id)

    await db.delete_cold_archive_search(ctx.session_id)

    result = await memory.backfill_cold_search_index(
        tenant_id="tenant1",
        user_id="user1",
        session_id=ctx.session_id,
    )
    assert result["indexed"] == 1
    rows = await db.select_many(
        "cold_archive_search",
        ["content"],
        where={"session_id": ctx.session_id},
    )
    assert len(rows) == 1
    assert "backfill keyword" in rows[0]["content"]
