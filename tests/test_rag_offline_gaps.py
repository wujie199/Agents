"""离线 RAG 补齐项：collection 统一、文件缓存、parent 扩展、manifest 反查。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.domain.context import RequestContext
from core.domain.evidence import Evidence, SourceType
from document.rag.application.chunking.parent_store import ParentChunkStore
from document.rag.application.embedding.collection import effective_collection_name
from document.rag.application.indexing.index_manifest import IndexManifest
from document.rag.application.retrieval.hybrid_pipeline import hybrid_retrieve
from document.rag.components.storage.embedding_file_cache import EmbeddingFileCacheAdapter
from document.rag.config import EmbeddingConfig, RagPipelineConfig


def test_effective_collection_name_versioned():
    cfg = RagPipelineConfig(
        collection_name="agent",
        model_version="bge-small-zh-v1.5",
        embedding=EmbeddingConfig(versioned_collection=True),
    )
    name = effective_collection_name(cfg)
    assert name.startswith("agent_")
    assert name != "agent"


def test_effective_collection_name_default():
    cfg = RagPipelineConfig(collection_name="agent")
    assert effective_collection_name(cfg) == "agent"


def test_embedding_file_cache_roundtrip(tmp_path):
    cache = EmbeddingFileCacheAdapter(tmp_path / "emb_cache")
    cache.set("emb:v1:abc", {"embedding": [1.0, 0.0]})
    assert cache.get("emb:v1:abc") == {"embedding": [1.0, 0.0]}
    assert cache.get("missing") is None


def test_manifest_find_by_doc_id(tmp_path):
    manifest = IndexManifest(tmp_path / "indexed_by_md5.json")
    manifest.register(
        "t1",
        "md5abc",
        doc_id="doc_xyz",
        source_path="/tmp/a.pdf",
        model_version="v1",
        config_hash="h1",
    )
    found = manifest.find_by_doc_id("t1", "doc_xyz")
    assert found is not None
    md5, entry = found
    assert md5 == "md5abc"
    assert entry["doc_id"] == "doc_xyz"
    assert manifest.find_by_doc_id("t1", "missing") is None


@pytest.mark.asyncio
async def test_hybrid_retrieve_expands_parent_context(tmp_path):
    parent_store = ParentChunkStore(tmp_path / "parents")
    collection = "test_coll"
    doc_id = "doc1"
    parent_id = "parent_1"
    parent_store.save(
        collection,
        doc_id,
        [],
    )
    # 直接写 parent json（绕过 ScoredChunk 构造）
    out_dir = tmp_path / "parents" / collection
    out_dir.mkdir(parents=True)
    (out_dir / f"{doc_id}.json").write_text(
        json.dumps(
            [
                {
                    "chunk_id": parent_id,
                    "content": "父 chunk 完整段落内容",
                    "heading_path": "第一章",
                    "child_ids": ["child_1"],
                    "metadata": {},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class _NoSearch:
        def similarity_search(self, *args, **kwargs):
            return []

    class _EmptyBm25:
        def search(self, *args, **kwargs):
            return []

    cfg = RagPipelineConfig(
        collection_name="test_coll",
        retrieval=__import__(
            "document.rag.config.retrieval", fromlist=["RetrievalConfig"]
        ).RetrievalConfig(
            enable_hybrid=False,
            enable_vector_search=True,
            enable_bm25_search=False,
            enable_rerank=False,
        ),
    )

    # 注入一条命中（绕过 vector search）
    from unittest.mock import AsyncMock, patch

    child_ev = Evidence(
        id="child_1",
        content="子 chunk 片段",
        source_type=SourceType.VECTOR,
        score=0.9,
        metadata={"doc_id": doc_id, "parent_id": parent_id, "tenant_id": "t1"},
    )

    with patch(
        "document.rag.application.retrieval.hybrid_pipeline._collect_hybrid_hits",
        new=AsyncMock(return_value=[child_ev]),
    ):
        ctx = RequestContext(
            tenant_id="t1",
            user_id="u",
            session_id="s",
            trace_id="tr",
            channel="test",
        )
        bundle = await hybrid_retrieve(
            "测试",
            ctx,
            vector_port=_NoSearch(),
            embedding_model=object(),
            bm25_index=_EmptyBm25(),
            rerank_model=None,
            config=cfg,
            parent_store=parent_store,
        )

    assert not bundle.empty
    parent_hits = [
        ev for ev in bundle.evidences if ev.metadata.get("chunk_role") == "parent"
    ]
    assert parent_hits
    assert "父 chunk" in parent_hits[0].content
