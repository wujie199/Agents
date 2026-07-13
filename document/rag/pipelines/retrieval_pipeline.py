"""检索管道 — 从 RAGPortAdapter 中提取的核心检索逻辑。

职责:
- 向量检索（embedding → similarity_search）
- 混合检索委托（BM25 + vector）
- 路由委托
- 缓存读/写
- 重排后处理
- ACL 过滤
- 结果组装
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.domain.context import RequestContext
from core.domain.evidence import DegradedReason, Evidence, EvidenceBundle, SourceType
from core.ports.rag.rerank import RerankPort
from core.ports.rag.embedding import EmbeddingPort
from core.ports.storage.cache import CachePort
from core.ports.storage.vector import VectorPort

from document.rag.config.embedding import EmbeddingConfig
from document.rag.config.pipeline import RagPipelineConfig
from document.rag.application.embedding.encoder import EmbeddingEncoder
from document.rag.application.retrieval.plan_codec import business_plan_to_router_override
from document.rag.application.retrieval.rerank_utils import apply_rerank
from document.rag.shared.evidence_helpers import bundle_from_cache_dict, bundle_to_cache_dict


@dataclass
class RetrievalPlan:
    primary_backend: str = "vector"
    secondary_backends: List[str] = field(default_factory=list)
    fusion_strategy: str = "rrf"
    cache_policy: str = "read_through"
    top_k: int = 10
    rerank_top_n: int = 5

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RetrievalPlan":
        if not data:
            return cls()
        return cls(
            primary_backend=data.get("primary", data.get("primary_backend", "vector")),
            top_k=int(data.get("top_k", 10)),
            rerank_top_n=int(data.get("rerank_top_n", 5)),
        )


def resolve_plan(
    plan: Optional[Any],
    default_top_k: int,
    default_rerank_top_n: int,
) -> RetrievalPlan:
    if plan is None:
        return RetrievalPlan(top_k=default_top_k, rerank_top_n=default_rerank_top_n)
    if isinstance(plan, RetrievalPlan):
        return plan
    if isinstance(plan, dict):
        resolved = RetrievalPlan.from_dict(plan)
        resolved.top_k = resolved.top_k or default_top_k
        resolved.rerank_top_n = resolved.rerank_top_n or default_rerank_top_n
        return resolved
    return RetrievalPlan(top_k=default_top_k, rerank_top_n=default_rerank_top_n)


def _hybrid_router_secondary(config: RagPipelineConfig) -> List[str]:
    if config.retrieval.enable_hybrid and config.retrieval.enable_bm25_search:
        return ["bm25"]
    return []


def router_plan_override(
    plan: Optional[Any],
    config: RagPipelineConfig,
    default_top_k: int,
    default_rerank_top_n: int,
) -> Optional[dict]:
    override = business_plan_to_router_override(plan, config)
    if override is not None:
        secondary = list(override.get("secondary") or [])
        for backend in _hybrid_router_secondary(config):
            if backend not in secondary:
                secondary.append(backend)
        if secondary:
            override["secondary"] = secondary
        if config.retrieval.enable_hybrid:
            override.setdefault(
                "fusion",
                config.retrieval.fusion_strategy or "weighted",
            )
        return override
    if not config.retrieval.auto_route:
        base: Dict[str, Any] = {
            "primary": config.retrieval.primary_backend,
            "top_k": default_top_k,
            "rerank_top_n": default_rerank_top_n,
        }
        secondary = _hybrid_router_secondary(config)
        if secondary:
            base["secondary"] = secondary
            base["fusion"] = config.retrieval.fusion_strategy or "weighted"
        return business_plan_to_router_override(base, config)
    return None


def get_cache_key(query: str, tenant_id: str) -> str:
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    return f"{tenant_id}:rag:qc:{query_hash}"


async def cache_get(cache_port: Optional[CachePort], key: str) -> Any:
    if not cache_port:
        return None
    value = cache_port.get(key)
    if asyncio.iscoroutine(value):
        return await value
    return value


async def cache_set(cache_port: Optional[CachePort], key: str, value: Any, ttl: Optional[int] = None) -> None:
    if not cache_port:
        return
    result = cache_port.set(key, value, ttl)
    if asyncio.iscoroutine(result):
        await result


async def cache_invalidate_pattern(cache_port: Optional[CachePort], pattern: str) -> None:
    if not cache_port or not hasattr(cache_port, "invalidate_pattern"):
        return
    result = cache_port.invalidate_pattern(pattern)
    if asyncio.iscoroutine(result):
        await result


async def get_embedding(
    text: str,
    embedding_model: EmbeddingPort,
    embedding_cfg: Optional[EmbeddingConfig] = None,
) -> List[float]:
    cfg = embedding_cfg or EmbeddingConfig()
    encoder = EmbeddingEncoder(embedding_model, cfg)
    return await encoder.encode_query(text)


async def get_embeddings(
    texts: List[str],
    embedding_model: EmbeddingPort,
    embedding_cfg: Optional[EmbeddingConfig] = None,
) -> List[List[float]]:
    cfg = embedding_cfg or EmbeddingConfig()
    encoder = EmbeddingEncoder(embedding_model, cfg)
    return await encoder.encode_queries(texts)


def check_acl(metadata: Dict, acl: Any) -> bool:
    if acl is None:
        return True
    doc_id = metadata.get("doc_id")
    if not doc_id or not hasattr(acl, "can_access_doc"):
        return True
    doc_ids = getattr(acl, "doc_ids", None)
    if doc_ids is not None and len(doc_ids) == 0:
        return True
    return acl.can_access_doc(doc_id)


async def vector_retrieve(
    query: str,
    context: RequestContext,
    vector_port: VectorPort,
    embedding_model: EmbeddingPort,
    collection: str,
    top_k: int,
    rerank_top_n: int,
    rerank_model: Optional[RerankPort],
    logger: logging.Logger,
    embedding_cfg: Optional[EmbeddingConfig] = None,
) -> EvidenceBundle:
    """纯向量检索 + 重排。"""
    try:
        query_vector = await get_embedding(
            query, embedding_model, embedding_cfg=embedding_cfg
        )
        search_results = await asyncio.to_thread(
            vector_port.similarity_search,
            collection,
            query_vector,
            top_k,
            {"tenant_id": context.tenant_id},
        )

        evidences: List[Evidence] = []
        for r in search_results:
            meta = r.metadata or {}
            if not check_acl(meta, context.acl):
                continue
            evidences.append(
                Evidence(
                    id=r.id,
                    content=r.content or "",
                    source_type=SourceType.VECTOR,
                    score=r.score,
                    citation=meta.get("source_path"),
                    metadata=meta,
                )
            )

        if rerank_model and len(evidences) > rerank_top_n:
            evidences = await apply_rerank(
                rerank_model, query, evidences, rerank_top_n, logger
            )

        evidences = evidences[:rerank_top_n]

        if not evidences:
            return EvidenceBundle.empty_bundle(
                reason=DegradedReason.PARTIAL_RESULTS,
                error_code="RAG_EMPTY",
                plan={"primary": "vector", "top_k": top_k, "collection": collection},
            )

        return EvidenceBundle(
            evidences=evidences,
            plan={"primary": "vector", "top_k": top_k, "collection": collection},
            empty=False,
        )

    except (RuntimeError, ConnectionError, TimeoutError, OSError, ValueError) as e:
        logger.error("RAG retrieval failed: %s", e)
        return EvidenceBundle.empty_bundle(
            reason=DegradedReason.VECTOR_UNAVAILABLE,
            error_code="RAG_001",
            plan={"error": str(e), "collection": collection},
        )
