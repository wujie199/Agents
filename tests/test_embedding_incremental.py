"""P2：chunk 增量、embedding 缓存读、DLQ、versioned collection。"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from core.ports.chunker import Chunk
from document.rag.application.embedding.collection import resolve_index_collection
from document.rag.application.embedding.dlq import append_embedding_dlq
from document.rag.application.embedding.fingerprint import chunk_embed_fingerprint
from document.rag.application.indexing.embedder import Embedder
from document.rag.application.indexing.index_manifest import IndexManifest
from document.rag.application.indexing.service import IndexService
from document.rag.components.embedding.mock import MockEmbeddingModel
from document.rag.config import EmbeddingConfig, RagPipelineConfig


class _CountingEmbedding:
    def __init__(self):
        self.calls = 0

    def embed(self, texts: List[str]) -> List[List[float]]:
        self.calls += 1
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)


class _MemoryCache:
    def __init__(self):
        self._store: Dict[str, Any] = {}

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        self._store[key] = value


class _TrackingVectorPort:
    def __init__(self):
        self.records: Dict[str, Any] = {}
        self.deleted_ids: List[str] = []
        self.upsert_calls = 0

    def upsert(self, collection, records):
        self.upsert_calls += 1
        for rec in records:
            self.records[rec.id] = rec
        return len(records)

    def delete_by_filter(self, collection, filter):
        doc_id = filter.get("doc_id")
        tenant_id = filter.get("tenant_id")
        to_del = [
            cid
            for cid, rec in self.records.items()
            if rec.metadata.get("doc_id") == doc_id
            and rec.metadata.get("tenant_id") == tenant_id
        ]
        for cid in to_del:
            del self.records[cid]
        return len(to_del)

    def delete_by_ids(self, collection, ids):
        n = 0
        for cid in ids:
            if cid in self.records:
                del self.records[cid]
                self.deleted_ids.append(cid)
                n += 1
        return n

    def list_ids_by_filter(self, collection, filter):
        doc_id = filter.get("doc_id")
        tenant_id = filter.get("tenant_id")
        return [
            cid
            for cid, rec in self.records.items()
            if rec.metadata.get("doc_id") == doc_id
            and rec.metadata.get("tenant_id") == tenant_id
        ]

    def count(self, collection):
        return len(self.records)


def test_chunk_embed_fingerprint_stable():
    fp1 = chunk_embed_fingerprint("hello", model_version="v1")
    fp2 = chunk_embed_fingerprint("hello", model_version="v1")
    fp3 = chunk_embed_fingerprint("hello", model_version="v2")
    assert fp1 == fp2
    assert fp1 != fp3


def test_resolve_versioned_collection():
    assert resolve_index_collection("agent", "bge-small-zh-v1.5", versioned=False) == "agent"
    name = resolve_index_collection("agent", "bge-small-zh-v1.5", versioned=True)
    assert name.startswith("agent_")
    assert "bge" in name


@pytest.mark.asyncio
async def test_embedder_skips_unchanged_fingerprints():
    model = _CountingEmbedding()
    cfg = EmbeddingConfig(enable_chunk_incremental=True, batch_size=8)
    embedder = Embedder(model, embedding_cfg=cfg, model_version="v1", enable_cache=False)
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", content="段落一", chunk_index=0),
        Chunk(chunk_id="c2", doc_id="d1", content="段落二", chunk_index=1),
    ]
    first = await embedder.embed_chunks(chunks, "t1")
    assert first.encoded_count == 2
    assert model.calls == 1

    prev = dict(first.fingerprints)
    second = await embedder.embed_chunks(chunks, "t1", previous_fingerprints=prev)
    assert second.skipped_unchanged == 2
    assert second.to_write == []
    assert second.encoded_count == 0
    assert model.calls == 1


@pytest.mark.asyncio
async def test_embedder_cache_read_avoids_encode():
    model = _CountingEmbedding()
    cache = _MemoryCache()
    cfg = EmbeddingConfig(enable_embedding_cache_read=True, batch_size=8)
    embedder = Embedder(
        model,
        embedding_cfg=cfg,
        cache_port=cache,
        model_version="v1",
        enable_cache=True,
    )
    chunks = [Chunk(chunk_id="c1", doc_id="d1", content="缓存测试", chunk_index=0)]

    await embedder.embed_chunks(chunks, "t1")
    assert model.calls == 1

    model.calls = 0
    again = await embedder.embed_chunks(chunks, "t1")
    assert model.calls == 0
    assert len(again.to_write) == 1


@pytest.mark.asyncio
async def test_index_service_deletes_orphan_chunks():
    vec = _TrackingVectorPort()
    vec.records["old_chunk"] = type(
        "R",
        (),
        {
            "id": "old_chunk",
            "metadata": {"doc_id": "doc1", "tenant_id": "t1"},
        },
    )()
    cfg = RagPipelineConfig(
        collection_name="test",
        chunk_size=200,
        chunk_overlap=0,
        embedding=EmbeddingConfig(enable_chunk_incremental=True),
    )
    service = IndexService(
        vector_port=vec,
        embedding_model=MockEmbeddingModel(dimension=8),
        config=cfg,
    )
    result = await service.index_document(
        doc_id="doc1",
        content="新内容只有一段",
        tenant_id="t1",
    )
    assert "old_chunk" in vec.deleted_ids
    assert result.chunks_deleted >= 1


@pytest.mark.asyncio
async def test_index_service_writes_dlq_on_failure(tmp_path):
    class _FailVector:
        def upsert(self, collection, records):
            raise RuntimeError("chroma down")

        def list_ids_by_filter(self, collection, filter):
            return []

        def count(self, collection):
            return 0

    dlq = tmp_path / "dlq.jsonl"
    cfg = RagPipelineConfig(
        collection_name="test",
        chunk_size=200,
        chunk_overlap=0,
        embedding=EmbeddingConfig(
            write_max_retries=1,
            dlq_path=str(dlq),
            enable_chunk_incremental=False,
        ),
    )
    service = IndexService(
        vector_port=_FailVector(),
        embedding_model=MockEmbeddingModel(dimension=8),
        config=cfg,
    )
    with pytest.raises(RuntimeError, match="chroma down"):
        await service.index_document(
            doc_id="doc1",
            content="写入失败测试",
            tenant_id="t1",
        )
    assert dlq.is_file()
    line = json.loads(dlq.read_text(encoding="utf-8").strip())
    assert line["doc_id"] == "doc1"
    assert line["expected_vectors"] >= 1


def test_manifest_persists_chunk_fingerprints(tmp_path):
    manifest = IndexManifest(tmp_path / "indexed_by_md5.json")
    fps = {"c1": "abc123", "c2": "def456"}
    manifest.register(
        "t1",
        "md5hex",
        doc_id="doc1",
        source_path="/tmp/a.pdf",
        model_version="v1",
        config_hash="hash1",
        chunk_fingerprints=fps,
    )
    entry = manifest.get_entry("t1", "md5hex")
    assert entry["chunk_fingerprints"] == fps


def test_append_embedding_dlq(tmp_path):
    path = tmp_path / "nested" / "dlq.jsonl"
    append_embedding_dlq(
        path,
        collection="agent",
        doc_id="d1",
        tenant_id="t1",
        expected=3,
        error="boom",
    )
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["error"] == "boom"
    assert record["expected_vectors"] == 3
