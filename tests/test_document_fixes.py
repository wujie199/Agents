"""document/ 缺陷修复回归测试。"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from core.domain.context import ACL, RequestContext
from core.domain.evidence import Evidence, SourceType
from document.rag.adapters.retrieval.bm25_local import LocalBm25Index
from document.rag.application.indexing.service import IndexService
from document.rag.application.retrieval.helpers import filter_evidences_by_acl
from document.rag.application.retrieval.hybrid_pipeline import (
    filter_evidences_by_tags,
    hybrid_retrieve,
)
from document.rag.application.retrieval.tag_filter import (
    chroma_safe_metadata,
    merge_tags_into_metadata,
    metadata_matches_tags,
    resolve_scenario_tags,
)
from document.rag.adapters.embedding.mock import MockEmbeddingModel
from document.rag.config import RagPipelineConfig, RetrievalConfig


class _FailingVectorPort:
    def upsert(self, collection, records):
        raise RuntimeError("vector write failed")

    def delete_by_filter(self, collection, filter):
        return 0

    def count(self, collection):
        return 0


class _TrackingBm25:
    def __init__(self):
        self.docs: dict[tuple[str, str], int] = {}

    def index_chunks(self, chunks, tenant_id, doc_id):
        self.docs[(tenant_id, doc_id)] = len(chunks)
        return len(chunks)

    def delete_by_doc_id(self, doc_id, tenant_id):
        return 1 if self.docs.pop((tenant_id, doc_id), None) is not None else 0


def test_bm25_rebuild_from_chroma_merges_other_tenants(tmp_path):
    idx = LocalBm25Index.for_collection(tmp_path, "agent")
    idx.index_chunks(
        [{"chunk_id": "b1", "content": "tenant b unique text", "metadata": {}}],
        tenant_id="t2",
        doc_id="doc_b",
    )

    mock_col = MagicMock()
    mock_col.get.return_value = {
        "ids": ["a1"],
        "documents": ["tenant a refreshed text"],
        "metadatas": [{"doc_id": "doc_a", "tenant_id": "t1"}],
    }
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_col

    with patch("chromadb.PersistentClient", return_value=mock_client):
        added = idx.rebuild_from_chroma(str(tmp_path / "chroma"), "agent", tenant_id="t1")

    assert added == 1
    assert idx.document_count == 2
    assert idx.search("unique text", tenant_id="t2")
    assert idx.search("refreshed", tenant_id="t1")


def test_index_service_rolls_back_bm25_when_vector_write_fails():
    bm25 = _TrackingBm25()
    service = IndexService(
        vector_port=_FailingVectorPort(),
        embedding_model=MockEmbeddingModel(dimension=8),
        config=RagPipelineConfig(collection_name="test", chunk_size=50, chunk_overlap=0),
        bm25_index=bm25,
    )

    with pytest.raises(RuntimeError, match="vector write failed"):
        asyncio.run(
            service.index_document(
                doc_id="doc1",
                content="hello world for rollback test",
                tenant_id="t1",
            )
        )
    assert ("t1", "doc1") not in bm25.docs


def test_filter_evidences_by_acl():
    evs = [
        Evidence(
            id="1",
            content="a",
            source_type=SourceType.VECTOR,
            score=1.0,
            metadata={"doc_id": "allowed"},
        ),
        Evidence(
            id="2",
            content="b",
            source_type=SourceType.VECTOR,
            score=0.5,
            metadata={"doc_id": "denied"},
        ),
    ]
    acl = ACL(doc_ids=frozenset({"allowed"}))
    out = filter_evidences_by_acl(evs, acl)
    assert [e.id for e in out] == ["1"]


@pytest.mark.asyncio
async def test_hybrid_retrieve_respects_acl(tmp_path):
    cfg = RagPipelineConfig(
        collection_name="agent",
        retrieval=RetrievalConfig(
            enable_hybrid=False,
            enable_vector_search=True,
            enable_bm25_search=False,
            enable_rerank=False,
        ),
    )

    class _VectorPort:
        def similarity_search(self, collection, query_vector, top_k, filter):
            return [
                MagicMock(
                    id="v1",
                    content="vector hit denied",
                    score=0.9,
                    metadata={"doc_id": "denied", "tenant_id": "t1"},
                ),
                MagicMock(
                    id="v2",
                    content="vector hit allowed",
                    score=0.8,
                    metadata={"doc_id": "allowed", "tenant_id": "t1"},
                ),
            ]

    ctx = RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        trace_id="tr1",
        channel="test",
        acl=ACL(doc_ids=frozenset({"allowed"})),
    )
    bundle = await hybrid_retrieve(
        "test query",
        ctx,
        vector_port=_VectorPort(),
        embedding_model=MockEmbeddingModel(dimension=8),
        bm25_index=LocalBm25Index.for_collection(tmp_path, "agent"),
        rerank_model=None,
        config=cfg,
    )
    assert [ev.id for ev in bundle.evidences] == ["v2"]


def test_build_one_document_plain_text_smoke(tmp_path):
    from document.build_rag_index import build_one_document
    from core.ports.index import IndexProfile
    from document.rag.config import EmbeddingConfig, IngestConfig, MetadataConfig

    txt = tmp_path / "sample.txt"
    txt.write_text("离线建库 smoke test 内容。", encoding="utf-8")
    data_dir = tmp_path / "data"
    cfg = RagPipelineConfig(
        collection_name="smoke",
        chunk_size=80,
        chunk_overlap=0,
        ingest=IngestConfig(mode="ocr_only", enable_cleaning=False),
        metadata=MetadataConfig(enabled=False),
        embedding=EmbeddingConfig(backend="mock"),
    )

    report = asyncio.run(
        build_one_document(
            txt,
            "doc_smoke",
            "t1",
            data_dir,
            str(tmp_path / "config"),
            IndexProfile.VECTOR_ONLY,
            cfg=cfg,
            skip_indexed=False,
            force_reindex=True,
        )
    )
    assert report.success, report.errors
    assert report.index is not None
    assert report.index.vectors_written >= 1


def test_metadata_matches_tags_any_and_all():
    meta = {"tags": ["合同", "法律"]}
    assert metadata_matches_tags(meta, ["合同"], match="any")
    assert metadata_matches_tags(meta, ["财务"], match="any") is False
    assert metadata_matches_tags(meta, ["合同", "法律"], match="all")
    assert metadata_matches_tags(meta, ["合同", "财务"], match="all") is False
    assert metadata_matches_tags(meta, [], match="any")
    assert metadata_matches_tags({"tags_csv": "FAQ,售后"}, ["FAQ"], match="any")


def test_chroma_safe_metadata_serializes_tags():
    safe = chroma_safe_metadata({"tags": ["合同", "法律"], "doc_id": "d1"})
    assert safe["tags"] == "合同,法律"
    assert safe["tags_csv"] == "合同,法律"
    assert safe["doc_id"] == "d1"


def test_merge_tags_into_metadata_dedupes():
    meta = merge_tags_into_metadata({"tags": ["FAQ"]}, ["售后", "FAQ"])
    assert meta["tags"] == ["FAQ", "售后"]


def test_resolve_scenario_tags(tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "scenarios.yml").write_text(
        "scenarios:\n  legal:\n    tags: [合同, 法律]\n    tag_match: all\n",
        encoding="utf-8",
    )
    tags, match = resolve_scenario_tags("legal", config_dir=str(cfg_dir))
    assert tags == ["合同", "法律"]
    assert match == "all"

    tags2, match2 = resolve_scenario_tags(
        "legal",
        config_dir=str(cfg_dir),
        explicit_tags=["发票"],
        tag_match="any",
    )
    assert tags2 == ["合同", "法律", "发票"]
    assert match2 == "any"


def test_filter_evidences_by_tags():
    evs = [
        Evidence(
            id="1",
            content="a",
            source_type=SourceType.VECTOR,
            score=1.0,
            metadata={"tags": ["合同"]},
        ),
        Evidence(
            id="2",
            content="b",
            source_type=SourceType.VECTOR,
            score=0.5,
            metadata={"tags": ["FAQ"]},
        ),
    ]
    out = filter_evidences_by_tags(evs, ["合同"], tag_match="any")
    assert [e.id for e in out] == ["1"]


@pytest.mark.asyncio
async def test_hybrid_retrieve_filters_by_tags(tmp_path):
    cfg = RagPipelineConfig(
        collection_name="agent",
        retrieval=RetrievalConfig(
            enable_hybrid=False,
            enable_vector_search=True,
            enable_bm25_search=False,
            enable_rerank=False,
        ),
    )

    class _VectorPort:
        def similarity_search(self, collection, query_vector, top_k, filter):
            return [
                MagicMock(
                    id="v1",
                    content="legal doc",
                    score=0.9,
                    metadata={"doc_id": "d1", "tenant_id": "t1", "tags": "合同"},
                ),
                MagicMock(
                    id="v2",
                    content="faq doc",
                    score=0.8,
                    metadata={"doc_id": "d2", "tenant_id": "t1", "tags": "FAQ"},
                ),
            ]

    ctx = RequestContext(
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        trace_id="tr1",
        channel="test",
    )
    bundle = await hybrid_retrieve(
        "test query",
        ctx,
        vector_port=_VectorPort(),
        embedding_model=MockEmbeddingModel(dimension=8),
        bm25_index=LocalBm25Index.for_collection(tmp_path, "agent"),
        rerank_model=None,
        config=cfg,
        tags=["合同"],
        tag_match="any",
    )
    assert [ev.id for ev in bundle.evidences] == ["v1"]
    assert bundle.plan.get("tags") == ["合同"]
