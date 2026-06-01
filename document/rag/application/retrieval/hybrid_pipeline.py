"""混合检索：向量 + BM25 → 加权融合 → Rerank。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

from core.domain.context import RequestContext
from core.domain.evidence import Evidence, EvidenceBundle, SourceType, DegradedReason
from document.rag.application.retrieval.helpers import search_results_to_evidences
from document.rag.application.retrieval.rerank_utils import apply_rerank
from document.rag.application.retrieval.router.fusion import FusionFactory
from document.rag.config import RagPipelineConfig

_log = logging.getLogger("document.rag.application.retrieval.hybrid_pipeline")


async def vector_search(
    vector_port: Any,
    embedding_model: Any,
    collection: str,
    query: str,
    top_k: int,
    tenant_id: str,
) -> List[Evidence]:
    if hasattr(embedding_model, "aembed"):
        vectors = await embedding_model.aembed([query])
    elif hasattr(embedding_model, "embed"):
        vectors = embedding_model.embed([query])
    else:
        raise RuntimeError("embedding model has no embed/aembed")
    query_vector = vectors[0]
    results = await asyncio.to_thread(
        vector_port.similarity_search,
        collection,
        query_vector,
        top_k,
        {"tenant_id": tenant_id},
    )
    evidences = search_results_to_evidences(results, SourceType.VECTOR)
    for ev in evidences:
        ev.metadata = {**ev.metadata, "retrieval_backend": "vector"}
    return evidences


def bm25_search(
    bm25_index: Any,
    query: str,
    top_k: int,
    tenant_id: str,
) -> List[Evidence]:
    raw = bm25_index.search(query, top_k=top_k, tenant_id=tenant_id)
    return search_results_to_evidences(raw, SourceType.VECTOR)


def dedupe_by_chunk_id(evidences: List[Evidence]) -> List[Evidence]:
    seen = set()
    out: List[Evidence] = []
    for ev in sorted(evidences, key=lambda e: e.score or 0.0, reverse=True):
        key = ev.id or ev.content[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


def _rerank_score(evidence: Evidence) -> float:
    raw = evidence.metadata.get("rerank_score")
    if raw is not None:
        return float(raw)
    return float(evidence.score or 0.0)


def filter_by_rerank_min_score(
    evidences: List[Evidence],
    min_score: Optional[float],
    *,
    rerank_applied: bool,
) -> List[Evidence]:
    """Rerank 后按 rerank 分数过滤（仅 rerank 生效时）。"""
    if min_score is None or not rerank_applied:
        return evidences
    return [ev for ev in evidences if _rerank_score(ev) > min_score]


async def hybrid_retrieve(
    query: str,
    context: RequestContext,
    *,
    vector_port: Any,
    embedding_model: Any,
    bm25_index: Any,
    rerank_model: Any,
    config: RagPipelineConfig,
    top_k: Optional[int] = None,
    rerank_n: Optional[int] = None,
    rerank_min_score: Optional[float] = None,
) -> EvidenceBundle:
    """向量 + BM25 加权融合，再 rerank。"""
    ret = config.retrieval
    vector_k = ret.vector_top_k or top_k or config.default_top_k
    bm25_k = ret.bm25_top_k or config.default_top_k
    fusion_n = ret.fusion_top_n or max(vector_k, bm25_k)
    rerank_n = rerank_n or ret.rerank_top_n or config.rerank_top_n
    tenant_id = context.tenant_id

    vector_hits: List[Evidence] = []
    bm25_hits: List[Evidence] = []

    if ret.enable_hybrid or ret.enable_vector_search:
        try:
            vector_hits = await vector_search(
                vector_port,
                embedding_model,
                config.collection_name,
                query,
                vector_k,
                tenant_id,
            )
        except Exception as exc:
            _log.warning("向量检索失败: %s", exc)

    if ret.enable_hybrid or ret.enable_bm25_search:
        try:
            bm25_hits = bm25_search(bm25_index, query, bm25_k, tenant_id)
        except Exception as exc:
            _log.warning("BM25 检索失败: %s", exc)

    if not vector_hits and not bm25_hits:
        return EvidenceBundle.empty_bundle(
            DegradedReason.ALL_BACKENDS_FAILED,
            "no_hits",
        )

    if ret.enable_hybrid and vector_hits and bm25_hits:
        strategy = FusionFactory.create(ret.fusion_strategy or "weighted")
        weights = list(ret.hybrid_weights or [0.5, 0.5])
        fused = strategy.fuse([vector_hits, bm25_hits], weights=weights)
    elif vector_hits:
        fused = vector_hits
    else:
        fused = bm25_hits

    fused = dedupe_by_chunk_id(fused)[:fusion_n]

    rerank_applied = False
    min_score = rerank_min_score if rerank_min_score is not None else ret.rerank_min_score
    if ret.enable_rerank and rerank_model and fused:
        fused = await apply_rerank(rerank_model, query, fused, rerank_n, _log)
        rerank_applied = True
    else:
        fused = fused[:rerank_n]

    fused = filter_by_rerank_min_score(
        fused,
        min_score,
        rerank_applied=rerank_applied,
    )

    return EvidenceBundle(
        evidences=fused,
        plan={
            "mode": "hybrid" if ret.enable_hybrid else "single",
            "vector_k": vector_k,
            "bm25_k": bm25_k,
            "fusion_strategy": ret.fusion_strategy,
            "hybrid_weights": ret.hybrid_weights,
            "collection": config.collection_name,
            "rerank_min_score": min_score if rerank_applied else None,
        },
        empty=len(fused) == 0,
    )
