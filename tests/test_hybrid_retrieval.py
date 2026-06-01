"""BM25 与混合检索单元测试（无本地模型）。"""

from pathlib import Path

import pytest

from core.domain.context import ACL, RequestContext
from core.domain.evidence import SourceType
from document.rag.adapters.retrieval.bm25_local import LocalBm25Index, tokenize
from document.rag.application.retrieval.hybrid_pipeline import (
    dedupe_by_chunk_id,
    filter_by_rerank_min_score,
)
from document.rag.application.retrieval.helpers import search_results_to_evidences
from document.rag.application.retrieval.router.fusion import WeightedFusion
from core.domain.evidence import Evidence


def test_tokenize_chinese_and_words():
    tokens = tokenize("扫地机器人 租赁 合同")
    assert "扫" in tokens
    assert "租" in tokens


def test_bm25_index_search(tmp_path: Path):
    idx = LocalBm25Index.for_collection(tmp_path, "agent")
    idx.index_chunks(
        [
            {"chunk_id": "c1", "content": "扫地机器人如何充电", "metadata": {}},
            {"chunk_id": "c2", "content": "租赁合同注意事项", "metadata": {}},
        ],
        tenant_id="default",
        doc_id="doc_a",
    )
    hits = idx.search("扫地机器人", top_k=5, tenant_id="default")
    assert hits
    assert hits[0]["id"] == "c1"
    assert hits[0]["score"] > 0


def test_weighted_fusion_prefers_both_lists():
    vector = [
        Evidence(id="a", content="v", source_type=SourceType.VECTOR, score=1.0),
        Evidence(id="b", content="vb", source_type=SourceType.VECTOR, score=0.2),
    ]
    bm25 = [
        Evidence(id="b", content="vb", source_type=SourceType.VECTOR, score=1.0),
        Evidence(id="c", content="bm", source_type=SourceType.VECTOR, score=0.5),
    ]
    fused = WeightedFusion().fuse([vector, bm25], weights=[0.5, 0.5])
    ids = [e.id for e in fused]
    assert "b" in ids
    assert len(ids) >= 2


def test_dedupe_by_chunk_id():
    evs = [
        Evidence(id="x", content="a", source_type=SourceType.VECTOR, score=0.5),
        Evidence(id="x", content="a", source_type=SourceType.VECTOR, score=0.9),
    ]
    out = dedupe_by_chunk_id(evs)
    assert len(out) == 1
    assert out[0].score == 0.9


def test_filter_by_rerank_min_score():
    evs = [
        Evidence(
            id="a",
            content="a",
            source_type=SourceType.VECTOR,
            score=0.9,
            metadata={"rerank_score": 0.9},
        ),
        Evidence(
            id="b",
            content="b",
            source_type=SourceType.VECTOR,
            score=0.5,
            metadata={"rerank_score": 0.5},
        ),
        Evidence(
            id="c",
            content="c",
            source_type=SourceType.VECTOR,
            score=0.8,
            metadata={"rerank_score": 0.8},
        ),
    ]
    out = filter_by_rerank_min_score(evs, 0.8, rerank_applied=True)
    assert [e.id for e in out] == ["a"]
    assert filter_by_rerank_min_score(evs, 0.8, rerank_applied=False) == evs
    assert filter_by_rerank_min_score(evs, None, rerank_applied=True) == evs

    raw = [{"id": "1", "content": "hello", "score": 0.8, "metadata": {"doc_id": "d1"}}]
    evs = search_results_to_evidences(raw, SourceType.VECTOR)
    assert len(evs) == 1
    assert evs[0].id == "1"
    assert evs[0].metadata["doc_id"] == "d1"
