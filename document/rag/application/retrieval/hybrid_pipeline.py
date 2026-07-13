"""混合检索：向量 + BM25 → 加权融合 → Rerank。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

from core.domain.context import RequestContext
from core.domain.evidence import Evidence, EvidenceBundle, SourceType, DegradedReason
from core.ports.rag import QueryRewritePort
from document.rag.application.retrieval.helpers import (
    filter_evidences_by_acl,
    search_results_to_evidences,
)
from document.rag.application.retrieval.query_intent import (
    apply_post_rerank_maintenance_routing,
    boost_maintenance_evidences,
    demote_faq_template_evidences,
    detect_maintenance_intent,
    resolve_retrieval_tags,
)
from document.rag.application.retrieval.rewrite.retrieval_quality import (
    retrieval_satisfactory,
)
from document.rag.application.retrieval.rewrite.rewrite_profile import (
    resolve_rewrite_profile,
)
from document.rag.application.embedding.collection import effective_collection_name
from document.rag.application.chunking.parent_resolver import (
    expand_evidences_with_parent_context,
)
from document.rag.application.retrieval.rerank_utils import apply_rerank
from document.rag.application.retrieval.router.fusion import FusionFactory
from document.rag.application.retrieval.tag_filter import metadata_matches_tags
from document.rag.config import RagPipelineConfig

_log = logging.getLogger("document.rag.application.retrieval.hybrid_pipeline")

_TAG_FETCH_MULTIPLIER = 4


async def _rewrite_queries(
    query: str,
    query_rewriter: Optional[QueryRewritePort],
    *,
    max_queries: int = 0,
    llm_mode: Optional[str] = None,
    profile: Optional[str] = None,
) -> List[str]:
    if query_rewriter is None or not getattr(query_rewriter, "is_enabled", lambda: False)():
        return [query]
    try:
        rewrite_fn = query_rewriter.rewrite
        if llm_mode is not None or profile is not None:
            kwargs: dict[str, Any] = {}
            if llm_mode is not None:
                kwargs["llm_mode"] = llm_mode
            if profile is not None:
                kwargs["profile"] = profile
            rewritten = await rewrite_fn(query, **kwargs)
        else:
            rewritten = await rewrite_fn(query)
        unique = list(dict.fromkeys(q.strip() for q in rewritten if q and q.strip()))
        queries = unique or [query]
    except (RuntimeError, ValueError, TypeError) as exc:
        _log.warning("Query rewrite failed: %s", exc)
        queries = [query]

    if max_queries > 0 and len(queries) > max_queries:
        _log.info(
            "Cap hybrid rewrite queries: %d -> %d",
            len(queries),
            max_queries,
        )
        queries = queries[:max_queries]
    return queries


async def _collect_hybrid_hits(
    queries: List[str],
    *,
    query: str,
    search_concurrency: int,
    use_intent_tags: bool,
    intent_tags: List[str],
    intent_match: str,
    search_tags: Optional[List[str]],
    search_match: str,
    search_kwargs: dict[str, Any],
) -> List[Evidence]:
    all_hits = await _hybrid_search_many(
        queries,
        concurrency=search_concurrency,
        tags=search_tags,
        tag_match=search_match,
        **search_kwargs,
    )
    if use_intent_tags and detect_maintenance_intent(query):
        tagged_hits = await _hybrid_search_many(
            queries[:2],
            concurrency=min(2, search_concurrency),
            tags=intent_tags,
            tag_match=intent_match,
            **search_kwargs,
        )
        if tagged_hits:
            all_hits = tagged_hits + all_hits
    return all_hits


async def _rerank_fused(
    query: str,
    all_hits: List[Evidence],
    *,
    fusion_n: int,
    len_queries: int,
    rerank_model: Any,
    rerank_n: int,
    min_score: Optional[float],
    ret: Any,
    rw_cfg: Any,
) -> tuple[List[Evidence], bool, List[Evidence]]:
    fused = dedupe_by_chunk_id(all_hits)[: fusion_n * max(1, len_queries)]
    if not fused:
        return [], False, []

    boost = float(getattr(rw_cfg, "maintenance_source_boost", 0.12) or 0.12)
    fused = boost_maintenance_evidences(fused, query, boost=boost)
    fused = fused[:fusion_n]

    rerank_applied = False
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
    post_boost = float(
        getattr(rw_cfg, "maintenance_post_rerank_boost", 0.18) or 0.18
    )
    faq_penalty = float(
        getattr(rw_cfg, "faq_non_maintenance_penalty", 0.12) or 0.12
    )
    fused = apply_post_rerank_maintenance_routing(
        fused,
        query,
        maintenance_boost=post_boost,
        faq_penalty=faq_penalty,
    )
    fused = demote_faq_template_evidences(fused, query)
    return fused, rerank_applied, fused


async def _hybrid_search_many(
    queries: List[str],
    *,
    concurrency: int,
    **search_kwargs: Any,
) -> List[Evidence]:
    if not queries:
        return []

    limit = max(1, concurrency)
    sem = asyncio.Semaphore(limit)

    async def _one(q: str) -> List[Evidence]:
        async with sem:
            return await _hybrid_search_once(q, **search_kwargs)

    batches = await asyncio.gather(*[_one(q) for q in queries], return_exceptions=True)
    all_hits: List[Evidence] = []
    for q, batch in zip(queries, batches):
        if isinstance(batch, Exception):
            _log.warning("Hybrid search failed for %r: %s", q[:60], batch)
            continue
        all_hits.extend(batch)
    return all_hits


def filter_evidences_by_tags(
    evidences: List[Evidence],
    tags: Optional[List[str]],
    *,
    tag_match: str = "any",
) -> List[Evidence]:
    if not tags:
        return evidences
    return [
        ev
        for ev in evidences
        if metadata_matches_tags(ev.metadata, tags, match=tag_match)
    ]


async def vector_search(
    vector_port: Any,
    embedding_model: Any,
    collection: str,
    query: str,
    top_k: int,
    tenant_id: str,
    *,
    tags: Optional[List[str]] = None,
    tag_match: str = "any",
) -> List[Evidence]:
    if hasattr(embedding_model, "aembed"):
        vectors = await embedding_model.aembed([query])
    elif hasattr(embedding_model, "embed"):
        vectors = embedding_model.embed([query])
    else:
        raise RuntimeError("embedding model has no embed/aembed")
    query_vector = vectors[0]
    fetch_k = top_k * _TAG_FETCH_MULTIPLIER if tags else top_k
    results = await asyncio.to_thread(
        vector_port.similarity_search,
        collection,
        query_vector,
        fetch_k,
        {"tenant_id": tenant_id},
    )
    evidences = search_results_to_evidences(results, SourceType.VECTOR)
    for ev in evidences:
        ev.metadata = {**ev.metadata, "retrieval_backend": "vector"}
    evidences = filter_evidences_by_tags(evidences, tags, tag_match=tag_match)
    return evidences[:top_k]


def bm25_search(
    bm25_index: Any,
    query: str,
    top_k: int,
    tenant_id: str,
    *,
    tags: Optional[List[str]] = None,
    tag_match: str = "any",
) -> List[Evidence]:
    fetch_k = top_k * _TAG_FETCH_MULTIPLIER if tags else top_k
    raw = bm25_index.search(query, top_k=fetch_k, tenant_id=tenant_id)
    evidences = search_results_to_evidences(raw, SourceType.VECTOR)
    evidences = filter_evidences_by_tags(evidences, tags, tag_match=tag_match)
    return evidences[:top_k]


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


async def _hybrid_search_once(
    query: str,
    *,
    vector_port: Any,
    embedding_model: Any,
    bm25_index: Any,
    config: RagPipelineConfig,
    collection: str,
    tenant_id: str,
    acl: Any,
    vector_k: int,
    bm25_k: int,
    fusion_n: int,
    tags: Optional[List[str]] = None,
    tag_match: str = "any",
) -> List[Evidence]:
    ret = config.retrieval
    vector_hits: List[Evidence] = []
    bm25_hits: List[Evidence] = []

    async def _fetch_vector() -> List[Evidence]:
        if not (ret.enable_hybrid or ret.enable_vector_search):
            return []
        try:
            return filter_evidences_by_acl(
                await vector_search(
                    vector_port,
                    embedding_model,
                    collection,
                    query,
                    vector_k,
                    tenant_id,
                    tags=tags,
                    tag_match=tag_match,
                ),
                acl,
            )
        except (RuntimeError, ConnectionError, TimeoutError, OSError, ValueError) as exc:
            _log.warning("向量检索失败: %s", exc)
            return []

    def _fetch_bm25() -> List[Evidence]:
        if not (ret.enable_hybrid or ret.enable_bm25_search):
            return []
        try:
            return filter_evidences_by_acl(
                bm25_search(
                    bm25_index,
                    query,
                    bm25_k,
                    tenant_id,
                    tags=tags,
                    tag_match=tag_match,
                ),
                acl,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            _log.warning("BM25 检索失败: %s", exc)
            return []

    need_vector = ret.enable_hybrid or ret.enable_vector_search
    need_bm25 = ret.enable_hybrid or ret.enable_bm25_search
    tasks: list[Any] = []
    task_labels: list[str] = []
    if need_vector:
        tasks.append(_fetch_vector())
        task_labels.append("vector")
    if need_bm25:
        tasks.append(asyncio.to_thread(_fetch_bm25))
        task_labels.append("bm25")

    if tasks:
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        for label, outcome in zip(task_labels, outcomes):
            if isinstance(outcome, Exception):
                _log.warning("%s 检索失败: %s", label, outcome)
                continue
            if label == "vector":
                vector_hits = outcome
            else:
                bm25_hits = outcome

    if not vector_hits and not bm25_hits:
        return []

    if ret.enable_hybrid and vector_hits and bm25_hits:
        strategy = FusionFactory.create(ret.fusion_strategy or "weighted")
        weights = list(ret.hybrid_weights or [0.5, 0.5])
        fused = strategy.fuse([vector_hits, bm25_hits], weights=weights)
    elif vector_hits:
        fused = vector_hits
    else:
        fused = bm25_hits

    return dedupe_by_chunk_id(fused)[:fusion_n]


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
    tags: Optional[List[str]] = None,
    tag_match: str = "any",
    query_rewriter: Optional[QueryRewritePort] = None,
    parent_store: Optional[Any] = None,
    expand_parents: bool = True,
) -> EvidenceBundle:
    """向量 + BM25 加权融合，再 rerank；支持规则改写多 query 与维护 metadata 软路由。"""
    ret = config.retrieval
    collection = effective_collection_name(config)
    vector_k = ret.vector_top_k or top_k or config.default_top_k
    bm25_k = ret.bm25_top_k or config.default_top_k
    fusion_n = ret.fusion_top_n or max(vector_k, bm25_k)
    rerank_n = rerank_n or ret.rerank_top_n or config.rerank_top_n
    tenant_id = context.tenant_id
    acl = context.acl

    rw_cfg = config.rewrite
    profile = resolve_rewrite_profile(query, default=rw_cfg.default_profile)
    policy = rw_cfg.profile_policy(profile)
    max_hybrid_q = policy.max_hybrid_queries
    search_concurrency = int(getattr(rw_cfg, "hybrid_search_concurrency", 4) or 4)

    stage1_llm = "always" if policy.llm == "always" else "never"
    queries = await _rewrite_queries(
        query,
        query_rewriter,
        max_queries=max_hybrid_q,
        llm_mode=stage1_llm,
        profile=profile,
    )
    rewrite_stage = 1

    intent_tags, intent_match = resolve_retrieval_tags(query)
    use_intent_tags = bool(intent_tags) and tags is None

    search_kwargs = dict(
        vector_port=vector_port,
        embedding_model=embedding_model,
        bm25_index=bm25_index,
        config=config,
        collection=collection,
        tenant_id=tenant_id,
        acl=acl,
        vector_k=vector_k,
        bm25_k=bm25_k,
        fusion_n=fusion_n,
    )
    search_tags = tags
    search_match = tag_match

    all_hits = await _collect_hybrid_hits(
        queries,
        query=query,
        search_concurrency=search_concurrency,
        use_intent_tags=use_intent_tags,
        intent_tags=intent_tags,
        intent_match=intent_match,
        search_tags=search_tags,
        search_match=search_match,
        search_kwargs=search_kwargs,
    )

    min_score = rerank_min_score if rerank_min_score is not None else ret.rerank_min_score
    fused, rerank_applied, _ = await _rerank_fused(
        query,
        all_hits,
        fusion_n=fusion_n,
        len_queries=len(queries),
        rerank_model=rerank_model,
        rerank_n=rerank_n,
        min_score=min_score,
        ret=ret,
        rw_cfg=rw_cfg,
    )

    if (
        rw_cfg.two_stage.enabled
        and policy.llm == "on_miss"
        and query_rewriter is not None
        and hasattr(query_rewriter, "expand_llm")
        and (
            not fused
            or not retrieval_satisfactory(
                query, profile, fused, rw_cfg.two_stage
            )
        )
    ):
        _log.info(
            "Two-stage retrieval: stage1 miss profile=%s top_score=%s",
            profile,
            fused[0].score if fused else None,
        )
        llm_extra = await query_rewriter.expand_llm(query)
        if llm_extra:
            merged = list(dict.fromkeys([*queries, *llm_extra]))
            queries = merged[:max_hybrid_q]
            all_hits2 = await _collect_hybrid_hits(
                queries,
                query=query,
                search_concurrency=search_concurrency,
                use_intent_tags=use_intent_tags,
                intent_tags=intent_tags,
                intent_match=intent_match,
                search_tags=search_tags,
                search_match=search_match,
                search_kwargs=search_kwargs,
            )
            fused2, rerank_applied2, _ = await _rerank_fused(
                query,
                all_hits2,
                fusion_n=fusion_n,
                len_queries=len(queries),
                rerank_model=rerank_model,
                rerank_n=rerank_n,
                min_score=min_score,
                ret=ret,
                rw_cfg=rw_cfg,
            )
            if fused2:
                fused = fused2
                rerank_applied = rerank_applied2
                rewrite_stage = 2

    if not fused:
        return EvidenceBundle.empty_bundle(
            DegradedReason.ALL_BACKENDS_FAILED,
            "no_hits",
        )

    if (
        expand_parents
        and parent_store is not None
        and config.chunk_pipeline.enable_parent_child
    ):
        fused = expand_evidences_with_parent_context(
            fused,
            collection,
            parent_store=parent_store,
        )

    return EvidenceBundle(
        evidences=fused,
        plan={
            "mode": "hybrid" if ret.enable_hybrid else "single",
            "vector_k": vector_k,
            "bm25_k": bm25_k,
            "fusion_strategy": ret.fusion_strategy,
            "hybrid_weights": ret.hybrid_weights,
            "collection": collection,
            "tags": tags or intent_tags or [],
            "tag_match": tag_match if tags else (intent_match if intent_tags else None),
            "rerank_min_score": min_score if rerank_applied else None,
            "rewrite_queries": queries,
            "rewrite_profile": profile,
            "rewrite_stage": rewrite_stage,
            "maintenance_intent": detect_maintenance_intent(query),
        },
        empty=len(fused) == 0,
    )
