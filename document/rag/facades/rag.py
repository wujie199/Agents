import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.domain.context import RequestContext
from core.domain.evidence import DegradedReason, Evidence, EvidenceBundle, SourceType
from core.ports.rag import RetrieveRequest
from core.ports.storage.cache import CachePort
from core.ports.storage.vector import VectorPort

from document.rag.config import RagPipelineConfig, load_rag_pipeline_config
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


class RAGPortAdapter:
    def __init__(
        self,
        vector_port: VectorPort,
        cache_port: Optional[CachePort] = None,
        embedding_model: Any = None,
        rerank_model: Any = None,
        config: Optional[RagPipelineConfig] = None,
        default_top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
        enable_cache: Optional[bool] = None,
        cache_ttl_seconds: Optional[int] = None,
        router: Optional[Any] = None,
    ):
        self._config = config or load_rag_pipeline_config()
        self._vector_port = vector_port
        self._cache_port = cache_port
        self._embedding_model = embedding_model
        self._rerank_model = rerank_model if self._config.retrieval.enable_rerank else None
        self._router = router
        self._collection = self._config.collection_name
        self._default_top_k = default_top_k or self._config.default_top_k
        self._rerank_top_n = rerank_top_n or self._config.rerank_top_n
        self._enable_cache = (
            enable_cache if enable_cache is not None else self._config.enable_cache
        )
        self._cache_ttl = cache_ttl_seconds or self._config.cache_ttl_seconds
        self._logger = logging.getLogger(__name__)
        self._cache_stats = {"hit": 0, "miss": 0}

    async def _get_embedding(self, text: str) -> List[float]:
        if self._embedding_model is None:
            raise RuntimeError("Embedding model not configured")
        if hasattr(self._embedding_model, "aembed"):
            embeddings = await self._embedding_model.aembed([text])
            return embeddings[0]
        if hasattr(self._embedding_model, "embed"):
            embeddings = self._embedding_model.embed([text])
            return embeddings[0]
        raise RuntimeError("Embedding model has no embed method")

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self._embedding_model is None:
            raise RuntimeError("Embedding model not configured")
        if hasattr(self._embedding_model, "aembed"):
            return await self._embedding_model.aembed(texts)
        if hasattr(self._embedding_model, "embed"):
            return self._embedding_model.embed(texts)
        raise RuntimeError("Embedding model has no embed method")

    def _get_cache_key(self, query: str, tenant_id: str) -> str:
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"{tenant_id}:rag:qc:{query_hash}"

    async def _cache_get(self, key: str) -> Any:
        if not self._cache_port:
            return None
        value = self._cache_port.get(key)
        if asyncio.iscoroutine(value):
            return await value
        return value

    async def _cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if not self._cache_port:
            return
        result = self._cache_port.set(key, value, ttl)
        if asyncio.iscoroutine(result):
            await result

    async def _cache_invalidate_pattern(self, pattern: str) -> None:
        if not self._cache_port or not hasattr(self._cache_port, "invalidate_pattern"):
            return
        result = self._cache_port.invalidate_pattern(pattern)
        if asyncio.iscoroutine(result):
            await result

    def get_cache_stats(self) -> dict[str, int]:
        return dict(self._cache_stats)

    async def _get_from_cache(self, key: str) -> Optional[EvidenceBundle]:
        if not self._enable_cache or self._cache_port is None:
            return None
        try:
            cached = await self._cache_get(key)
            if cached:
                self._cache_stats["hit"] += 1
                return bundle_from_cache_dict(cached)
            self._cache_stats["miss"] += 1
        except Exception as e:
            self._cache_stats["miss"] += 1
            self._logger.warning("Cache get failed: %s", e)
        return None

    async def _set_cache(self, key: str, bundle: EvidenceBundle) -> None:
        if not self._enable_cache or self._cache_port is None:
            return
        try:
            await self._cache_set(key, bundle_to_cache_dict(bundle), self._cache_ttl)
        except Exception as e:
            self._logger.warning("Cache set failed: %s", e)

    def _resolve_plan(self, plan: Optional[Any]) -> RetrievalPlan:
        if plan is None:
            return RetrievalPlan(
                top_k=self._default_top_k,
                rerank_top_n=self._rerank_top_n,
            )
        if isinstance(plan, RetrievalPlan):
            return plan
        if isinstance(plan, dict):
            resolved = RetrievalPlan.from_dict(plan)
            resolved.top_k = resolved.top_k or self._default_top_k
            resolved.rerank_top_n = resolved.rerank_top_n or self._rerank_top_n
            return resolved
        return RetrievalPlan(top_k=self._default_top_k, rerank_top_n=self._rerank_top_n)

    def _router_plan_override(self, plan: Optional[Any]) -> Optional[dict]:
        override = business_plan_to_router_override(plan, self._config)
        if override is not None:
            return override
        if not self._config.retrieval.auto_route:
            return business_plan_to_router_override(
                {
                    "primary": self._config.retrieval.primary_backend,
                    "top_k": self._default_top_k,
                    "rerank_top_n": self._rerank_top_n,
                },
                self._config,
            )
        return None

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
            except Exception:
                pass

        if self._router is not None and self._config.retrieval.enable_router:
            _rag_trace(True, "delegated_retrieval_router")
            return await self._router.route_and_retrieve(
                query,
                context,
                self._router_plan_override(plan),
            )

        resolved = self._resolve_plan(plan)

        cache_key = self._get_cache_key(query, context.tenant_id)
        cached_result = await self._get_from_cache(cache_key)
        if cached_result:
            _rag_trace(True, "redis_cache_hit", cache_key=cache_key)
            return cached_result
        _rag_trace(True, "vector_search_start", cache_key=cache_key)

        try:
            query_vector = await self._get_embedding(query)
            search_results = await asyncio.to_thread(
                self._vector_port.similarity_search,
                self._collection,
                query_vector,
                resolved.top_k,
                {"tenant_id": context.tenant_id},
            )

            evidences: List[Evidence] = []
            for r in search_results:
                meta = r.metadata or {}
                if not self._check_acl(meta, context.acl):
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

            if self._rerank_model and len(evidences) > resolved.rerank_top_n:
                evidences = await apply_rerank(
                    self._rerank_model,
                    query,
                    evidences,
                    resolved.rerank_top_n,
                    self._logger,
                )

            evidences = evidences[: resolved.rerank_top_n]

            if not evidences:
                bundle = EvidenceBundle.empty_bundle(
                    reason=DegradedReason.PARTIAL_RESULTS,
                    error_code="RAG_EMPTY",
                    plan={
                        "primary": resolved.primary_backend,
                        "top_k": resolved.top_k,
                        "collection": self._collection,
                    },
                )
            else:
                bundle = EvidenceBundle(
                    evidences=evidences,
                    plan={
                        "primary": resolved.primary_backend,
                        "top_k": resolved.top_k,
                        "collection": self._collection,
                    },
                    empty=False,
                )

            await self._set_cache(cache_key, bundle)
            return bundle

        except Exception as e:
            self._logger.error("RAG retrieval failed: %s", e)
            return EvidenceBundle.empty_bundle(
                reason=DegradedReason.VECTOR_UNAVAILABLE,
                error_code="RAG_001",
                plan={"error": str(e), "collection": self._collection},
            )

    async def route_and_retrieve_batch(
        self,
        requests: List[RetrieveRequest],
        context: RequestContext,
        plan: Optional[Any] = None,
    ) -> List[EvidenceBundle]:
        if self._router is not None and self._config.retrieval.enable_router:
            results = []
            default_override = self._router_plan_override(plan)
            for req in requests:
                override = business_plan_to_router_override(
                    req.plan_override, self._config
                ) or default_override
                results.append(
                    await self._router.route_and_retrieve(
                        req.query, context, override
                    )
                )
            return results

        resolved = self._resolve_plan(plan)
        queries = [r.query for r in requests]

        try:
            query_vectors = await self._get_embeddings(queries)
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
                        if self._check_acl(r.metadata or {}, context.acl)
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
                        results.append(
                            EvidenceBundle(
                                evidences=evidences[: resolved.rerank_top_n],
                                plan=plan_override,
                                empty=False,
                            )
                        )
                except Exception as e:
                    self._logger.error("Batch retrieval failed: %s", e)
                    results.append(
                        EvidenceBundle.empty_bundle(
                            reason=DegradedReason.VECTOR_UNAVAILABLE,
                            error_code="RAG_002",
                            plan=plan_override,
                        )
                    )
            return results

        except Exception as e:
            self._logger.error("Batch embedding failed: %s", e)
            return [
                EvidenceBundle.empty_bundle(
                    reason=DegradedReason.ALL_BACKENDS_FAILED,
                    error_code="RAG_003",
                )
                for _ in requests
            ]

    def _check_acl(self, metadata: Dict, acl: Any) -> bool:
        if acl is None:
            return True
        doc_id = metadata.get("doc_id")
        if not doc_id or not hasattr(acl, "can_access_doc"):
            return True
        doc_ids = getattr(acl, "doc_ids", None)
        if doc_ids is not None and len(doc_ids) == 0:
            return True
        return acl.can_access_doc(doc_id)

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
            await self._cache_invalidate_pattern(f"{tenant_id}:rag:qc:*")
        except Exception as e:
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
        except Exception as e:
            health_info["vector_port"] = f"error: {e}"
            health_info["status"] = "degraded"

        if self._cache_port and hasattr(self._cache_port, "health"):
            try:
                cache_health = self._cache_port.health()
                health_info["cache_port"] = cache_health.get("status", "unknown")
            except Exception as e:
                health_info["cache_port"] = f"error: {e}"

        return health_info
