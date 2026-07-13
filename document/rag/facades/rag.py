"""RAG Port 适配器（薄层）— 委托 pipelines/retrieval_pipeline。"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from core.domain.context import RequestContext
from core.domain.evidence import DegradedReason, Evidence, EvidenceBundle, SourceType
from core.ports.rag import RetrieveRequest, RerankPort, QueryRewritePort
from core.ports.rag.embedding import EmbeddingPort
from core.ports.storage.cache import CachePort
from core.ports.storage.vector import VectorPort

from document.rag.application.embedding.collection import effective_collection_name
from document.rag.config.pipeline import RagPipelineConfig
from document.rag.application.retrieval.rerank_utils import apply_rerank
from document.rag.pipelines.retrieval_pipeline import (
    RetrievalPlan,
    cache_get,
    cache_invalidate_pattern,
    cache_set,
    check_acl,
    get_cache_key,
    get_embedding,
    get_embeddings,
    resolve_plan,
    router_plan_override,
    vector_retrieve,
)
from document.rag.shared.evidence_helpers import bundle_from_cache_dict, bundle_to_cache_dict


class RAGPortAdapter:
    def __init__(
        self,
        vector_port: VectorPort,
        config: RagPipelineConfig,
        cache_port: Optional[CachePort] = None,
        embedding_model: Optional[EmbeddingPort] = None,
        rerank_model: Optional[RerankPort] = None,
        default_top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
        enable_cache: Optional[bool] = None,
        cache_ttl_seconds: Optional[int] = None,
        router: Optional[Any] = None,
        bm25_index: Optional[Any] = None,
        query_rewriter: Optional[QueryRewritePort] = None,
    ):
        self._config = config
        self._vector_port = vector_port
        self._cache_port = cache_port
        self._embedding_model = embedding_model
        self._rerank_model = rerank_model if self._config.retrieval.enable_rerank else None
        self._router = router
        self._bm25_index = bm25_index
        self._query_rewriter = query_rewriter
        self._collection = effective_collection_name(self._config)
        self._default_top_k = default_top_k or self._config.default_top_k
        self._rerank_top_n = rerank_top_n or self._config.rerank_top_n
        self._enable_cache = (
            enable_cache if enable_cache is not None else self._config.enable_cache
        )
        self._cache_ttl = cache_ttl_seconds or self._config.cache_ttl_seconds
        self._logger = logging.getLogger(__name__)
        self._cache_stats = {"hit": 0, "miss": 0}

    def get_cache_stats(self) -> dict[str, int]:
        return dict(self._cache_stats)

    async def _get_from_cache(self, key: str) -> Optional[EvidenceBundle]:
        if not self._enable_cache or self._cache_port is None:
            return None
        try:
            cached = await cache_get(self._cache_port, key)
            if cached:
                self._cache_stats["hit"] += 1
                return bundle_from_cache_dict(cached)
            self._cache_stats["miss"] += 1
        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError, TypeError) as e:
            self._cache_stats["miss"] += 1
            self._logger.warning("Cache get failed: %s", e)
        return None

    async def _set_cache(self, key: str, bundle: EvidenceBundle) -> None:
        if not self._enable_cache or self._cache_port is None:
            return
        try:
            await cache_set(self._cache_port, key, bundle_to_cache_dict(bundle), self._cache_ttl)
        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError, TypeError) as e:
            self._logger.warning("Cache set failed: %s", e)

    async def route_and_retrieve(
        self,
        query: str,
        context: RequestContext,
        plan: Optional[Any] = None,
    ) -> EvidenceBundle:
        def _rag_trace(triggered: bool, reason: str, **extra: Any) -> None:
            try:
                from app.agents.memory.memory_runtime_debug import trace_layer_trigger

                trace_layer_trigger(
                    None,
                    "RAG",
                    "route_and_retrieve",
                    triggered,
                    reason,
                    data={
                        "tenant_id": context.tenant_id,
                        "query_preview": (query or "")[:80],
                        "enable_router": self._config.retrieval.enable_router,
                        **extra,
                    },
                    run_id=context.session_id,
                )
            except (ImportError, AttributeError, RuntimeError):
                pass

        # 1. 混合检索（优先于 router，确保 enable_hybrid 时走 hybrid_retrieve + Redis 缓存）
        resolved = resolve_plan(plan, self._default_top_k, self._rerank_top_n)
        use_hybrid = self._bm25_index is not None and (
            self._config.retrieval.enable_hybrid
            or self._config.retrieval.enable_bm25_search
        )
        if use_hybrid:
            from document.rag.application.retrieval.hybrid_pipeline import (
                _rewrite_queries,
            )

            rewrite_queries = await _rewrite_queries(query, self._query_rewriter)
            cache_payload = "|".join(sorted(rewrite_queries))
            cache_key = get_cache_key(cache_payload, context.tenant_id)
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                _rag_trace(True, "redis_cache_hit", cache_key=cache_key)
                return cached_result
            from document.rag.application.retrieval.hybrid_pipeline import hybrid_retrieve
            _rag_trace(True, "hybrid_retrieve", cache_key=cache_key)
            bundle = await hybrid_retrieve(
                query,
                context,
                vector_port=self._vector_port,
                embedding_model=self._embedding_model,
                bm25_index=self._bm25_index,
                rerank_model=self._rerank_model,
                config=self._config,
                top_k=resolved.top_k,
                rerank_n=resolved.rerank_top_n,
                query_rewriter=self._query_rewriter,
            )
            await self._set_cache(cache_key, bundle)
            return bundle

        # 2. 路由委托
        if self._router is not None and self._config.retrieval.enable_router:
            _rag_trace(True, "delegated_retrieval_router")
            return await self._router.route_and_retrieve(
                query,
                context,
                router_plan_override(
                    plan, self._config, self._default_top_k, self._rerank_top_n
                ),
            )

        # 3. 缓存读取
        cache_key = get_cache_key(query, context.tenant_id)
        cached_result = await self._get_from_cache(cache_key)
        if cached_result:
            _rag_trace(True, "redis_cache_hit", cache_key=cache_key)
            return cached_result
        _rag_trace(True, "vector_search_start", cache_key=cache_key)

        # 4. 向量检索
        bundle = await vector_retrieve(
            query, context,
            vector_port=self._vector_port,
            embedding_model=self._embedding_model,
            collection=self._collection,
            top_k=resolved.top_k,
            rerank_top_n=resolved.rerank_top_n,
            rerank_model=self._rerank_model,
            logger=self._logger,
            embedding_cfg=self._config.embedding,
        )

        # 5. 缓存写入
        await self._set_cache(cache_key, bundle)
        return bundle

    async def route_and_retrieve_batch(
        self,
        requests: List[RetrieveRequest],
        context: RequestContext,
        plan: Optional[Any] = None,
    ) -> List[EvidenceBundle]:
        if self._router is not None and self._config.retrieval.enable_router:
            results = []
            default_override = router_plan_override(
                plan, self._config, self._default_top_k, self._rerank_top_n
            )
            for req in requests:
                from document.rag.application.retrieval.plan_codec import business_plan_to_router_override
                override = business_plan_to_router_override(
                    req.plan_override, self._config
                ) or default_override
                results.append(
                    await self._router.route_and_retrieve(
                        req.query, context, override
                    )
                )
            return results

        resolved = resolve_plan(plan, self._default_top_k, self._rerank_top_n)
        use_hybrid = self._bm25_index is not None and (
            self._config.retrieval.enable_hybrid
            or self._config.retrieval.enable_bm25_search
        )
        if use_hybrid:
            results: List[EvidenceBundle] = []
            for req in requests:
                plan_override = req.plan_override or {
                    "primary": resolved.primary_backend,
                    "top_k": resolved.top_k,
                    "rerank_top_n": resolved.rerank_top_n,
                }
                try:
                    bundle = await self.route_and_retrieve(
                        req.query, context, plan=plan_override
                    )
                    results.append(bundle)
                except (RuntimeError, ConnectionError, TimeoutError, OSError, ValueError) as e:
                    self._logger.error("Batch hybrid retrieval failed: %s", e)
                    results.append(
                        EvidenceBundle.empty_bundle(
                            reason=DegradedReason.ALL_BACKENDS_FAILED,
                            error_code="RAG_003",
                            plan=plan_override,
                        )
                    )
            return results

        queries = [r.query for r in requests]

        try:
            query_vectors = await get_embeddings(
                queries, self._embedding_model, embedding_cfg=self._config.embedding
            )
            results: List[EvidenceBundle] = []

            for request, query_vector in zip(requests, query_vectors):
                plan_override = request.plan_override or {
                    "primary": resolved.primary_backend,
                    "top_k": resolved.top_k,
                }
                try:
                    search_results = await asyncio.to_thread(
                        self._vector_port.similarity_search,
                        self._collection,
                        query_vector,
                        resolved.top_k,
                        {"tenant_id": context.tenant_id},
                    )
                    evidences = [
                        Evidence(
                            id=r.id,
                            content=r.content or "",
                            source_type=SourceType.VECTOR,
                            score=r.score,
                            citation=(r.metadata or {}).get("source_path"),
                            metadata=r.metadata or {},
                        )
                        for r in search_results
                        if check_acl(r.metadata or {}, context.acl)
                    ]
                    if not evidences:
                        results.append(
                            EvidenceBundle.empty_bundle(
                                reason=DegradedReason.PARTIAL_RESULTS,
                                error_code="RAG_EMPTY",
                                plan=plan_override,
                            )
                        )
                    else:
                        top_n = resolved.rerank_top_n
                        if self._rerank_model and len(evidences) > top_n:
                            evidences = await apply_rerank(
                                self._rerank_model,
                                request.query,
                                evidences,
                                top_n,
                                self._logger,
                            )
                        results.append(
                            EvidenceBundle(
                                evidences=evidences[:top_n],
                                plan=plan_override,
                                empty=False,
                            )
                        )
                except (RuntimeError, ConnectionError, TimeoutError, OSError, ValueError) as e:
                    self._logger.error("Batch retrieval failed: %s", e)
                    results.append(
                        EvidenceBundle.empty_bundle(
                            reason=DegradedReason.VECTOR_UNAVAILABLE,
                            error_code="RAG_002",
                            plan=plan_override,
                        )
                    )
            return results

        except (RuntimeError, ConnectionError, TimeoutError, OSError, ValueError) as e:
            self._logger.error("Batch embedding failed: %s", e)
            return [
                EvidenceBundle.empty_bundle(
                    reason=DegradedReason.ALL_BACKENDS_FAILED,
                    error_code="RAG_003",
                )
                for _ in requests
            ]

    async def invalidate_document(self, doc_id: str, tenant_id: str) -> None:
        try:
            deleted = await asyncio.to_thread(
                self._vector_port.delete_by_doc_id,
                self._collection,
                doc_id,
                tenant_id,
            )
            self._logger.info(
                "Invalidated document %s in %s: %d chunks",
                doc_id,
                self._collection,
                deleted,
            )
            await cache_invalidate_pattern(self._cache_port, f"{tenant_id}:rag:qc:*")
        except (RuntimeError, ConnectionError, TimeoutError, OSError, ValueError) as e:
            self._logger.error("Document invalidation failed: %s", e)

    async def health(self) -> dict:
        health_info = {
            "status": "healthy",
            "collection": self._collection,
            "vector_port": "unknown",
            "cache_port": "disabled",
        }
        try:
            vector_health = self._vector_port.health()
            health_info["vector_port"] = vector_health.get("status", "unknown")
        except (ConnectionError, TimeoutError, OSError, RuntimeError, AttributeError) as e:
            health_info["vector_port"] = f"error: {e}"
            health_info["status"] = "degraded"

        if self._cache_port and hasattr(self._cache_port, "health"):
            try:
                cache_health = self._cache_port.health()
                health_info["cache_port"] = cache_health.get("status", "unknown")
            except (ConnectionError, TimeoutError, OSError, RuntimeError, AttributeError) as e:
                health_info["cache_port"] = f"error: {e}"

        return health_info
