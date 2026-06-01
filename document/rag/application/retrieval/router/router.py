import asyncio
import hashlib
import json
from typing import Optional, List, Dict, Any
import logging

from core.domain.context import RequestContext
from core.domain.evidence import (
    DegradedReason,
    Evidence,
    EvidenceBundle,
    SourceType,
)
from document.rag.application.retrieval.router.classifier import QueryClassifier, ClassificationResult
from document.rag.application.retrieval.router.rules import RoutingRules, RetrievalPlan, BackendType
from document.rag.application.retrieval.router.fusion import FusionFactory
from document.rag.application.retrieval.helpers import (
    cache_get,
    cache_set,
    check_acl,
    vector_similarity_search,
)
from document.rag.application.indexing.document_store import RagDocumentStore
from document.rag.application.retrieval.rerank_utils import apply_rerank
from document.rag.shared.evidence_helpers import bundle_from_cache_dict, bundle_to_cache_dict


class RetrievalRouter:
    """
    
    æ ¸å¿æµç¨ï¼?    1. æ¥è¯¢åç±»ï¼query_typeï¼?    2. è§åè·¯ç±ï¼çæ?retrieval_planï¼?    3. æ§è¡è®¡åï¼å¹¶è¡?çº§èè°ç¨ååç«¯ï¼
    4. ç»æèåï¼RRF/weighted/cascadeï¼?    5. ç¼å­å¤ç
    """
    
    def __init__(
        self,
        classifier: Optional[QueryClassifier] = None,
        rules: Optional[RoutingRules] = None,
        cache_port: Optional[Any] = None,
        vector_port: Optional[Any] = None,
        sql_port: Optional[Any] = None,
        graph_port: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        rerank_model: Optional[Any] = None,
        query_rewriter: Optional[Any] = None,
        collection_name: str = "agent",
        enable_cache: bool = True,
        cache_ttl: int = 900,
        enable_graph: bool = False,
        enable_sql: bool = False,
        enable_rerank: bool = True,
        default_top_k: int = 10,
        default_rerank_n: int = 5,
    ):
        self._classifier = classifier or QueryClassifier()
        self._rules = rules or RoutingRules(
            enable_graph=enable_graph,
            enable_sql=enable_sql,
        )
        self._collection = collection_name
        self._cache_port = cache_port
        self._vector_port = vector_port
        self._sql_port = sql_port
        self._graph_port = graph_port
        self._embedding_model = embedding_model
        self._rerank_model = rerank_model if enable_rerank else None
        self._query_rewriter = query_rewriter
        self._doc_store = RagDocumentStore(sql_port) if sql_port and enable_sql else None
        self._enable_cache = enable_cache
        self._cache_ttl = cache_ttl
        self._default_top_k = default_top_k
        self._default_rerank_n = default_rerank_n
        self._logger = logging.getLogger("rag.router")
    
    async def route_and_retrieve(
        self,
        query: str,
        context: RequestContext,
        plan_override: Optional[Dict[str, Any]] = None
    ) -> EvidenceBundle:
        classification = self._classifier.classify(query)
        
        self._logger.info(
            f"Query classified as {classification.query_type.value} "
            f"with confidence {classification.confidence:.2f}"
        )
        
        if plan_override:
            plan = RetrievalPlan(
                primary=BackendType(plan_override.get("primary", "vector")),
                secondary=[BackendType(b) for b in plan_override.get("secondary", [])],
                fusion=plan_override.get("fusion", "rrf"),
                cache_policy=plan_override.get("cache_policy", "read_through"),
                top_k=plan_override.get("top_k", 10),
                rerank_top_n=plan_override.get("rerank_top_n", 5),
                graph_hop=plan_override.get("graph_hop", 0),
            )
        else:
            plan = self._rules.route(classification, {"tenant_id": context.tenant_id})
        
        self._logger.info(f"Retrieval plan: {plan.to_dict()}")
        
        queries = await self._rewrite_queries(query)

        if plan.cache_policy != "no_cache":
            cache_key = self._get_cache_key(queries, context.tenant_id)
            cached = await self._get_from_cache(cache_key)
            if cached:
                self._logger.info("Cache hit, returning cached results")
                return cached

        results = await self._execute_plan_multi(queries, plan, context, classification)

        fusion = FusionFactory.create(plan.fusion)
        fused_evidences = fusion.fuse(results)

        rerank_n = plan.rerank_top_n or self._default_rerank_n
        if rerank_n > 0 and self._rerank_model:
            fused_evidences = await apply_rerank(
                self._rerank_model,
                query,
                fused_evidences,
                rerank_n,
                self._logger,
            )
        
        final_evidences = fused_evidences[:plan.top_k]

        plan_dict = {**plan.to_dict(), "collection": self._collection}
        if not final_evidences:
            bundle = EvidenceBundle.empty_bundle(
                reason=DegradedReason.PARTIAL_RESULTS,
                error_code="RAG_EMPTY",
                plan=plan_dict,
            )
        else:
            bundle = EvidenceBundle(
                evidences=final_evidences,
                plan=plan_dict,
                empty=False,
            )

        if plan.cache_policy == "read_through":
            cache_key = self._get_cache_key(queries, context.tenant_id)
            await self._set_cache(cache_key, bundle)

        return bundle

    async def _rewrite_queries(self, query: str) -> List[str]:
        if self._query_rewriter is None or not getattr(
            self._query_rewriter, "is_enabled", lambda: False
        )():
            return [query]
        try:
            rewritten = await self._query_rewriter.rewrite(query)
            unique = list(dict.fromkeys(q.strip() for q in rewritten if q and q.strip()))
            return unique or [query]
        except Exception as exc:
            self._logger.warning("Query rewrite failed: %s", exc)
            return [query]

    async def _execute_plan_multi(
        self,
        queries: List[str],
        plan: RetrievalPlan,
        context: RequestContext,
        classification: ClassificationResult,
    ) -> List[List[Evidence]]:
        backends = [plan.primary] + plan.secondary
        merged: Dict[str, List[Evidence]] = {b.value: [] for b in backends}

        for q in queries:
            per_query = await self._execute_plan(q, plan, context, classification)
            for backend, chunk in zip(backends, per_query):
                merged[backend.value].extend(chunk)

        return [merged[b.value] for b in backends]
    
    async def route_and_retrieve_batch(
        self,
        queries: List[str],
        context: RequestContext,
        plan_override: Optional[Dict[str, Any]] = None
    ) -> List[EvidenceBundle]:
        tasks = [
            self.route_and_retrieve(query, context, plan_override)
            for query in queries
        ]
        
        return await asyncio.gather(*tasks)
    
    async def _execute_plan(
        self,
        query: str,
        plan: RetrievalPlan,
        context: RequestContext,
        classification: ClassificationResult
    ) -> List[List[Evidence]]:
        backends = [plan.primary] + plan.secondary
        
        if plan.order.value == "parallel":
            tasks = [
                self._retrieve_from_backend(query, backend, plan, context, classification)
                for backend in backends
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self._logger.error(f"Backend {backends[i].value} failed: {result}")
                    processed_results.append([])
                else:
                    processed_results.append(result)
            
            return processed_results
        
        else:
            results = []
            for backend in backends:
                try:
                    result = await self._retrieve_from_backend(
                        query, backend, plan, context, classification
                    )
                    results.append(result)
                except Exception as e:
                    self._logger.error(f"Backend {backend.value} failed: {e}")
                    results.append([])
            
            return results
    
    async def _retrieve_from_backend(
        self,
        query: str,
        backend: BackendType,
        plan: RetrievalPlan,
        context: RequestContext,
        classification: ClassificationResult
    ) -> List[Evidence]:
        if backend == BackendType.REDIS_CACHE:
            return await self._retrieve_from_cache(query, context)
        
        elif backend == BackendType.VECTOR:
            return await self._retrieve_from_vector(query, plan, context)
        
        elif backend == BackendType.SQL:
            return await self._retrieve_from_sql(query, classification, context)
        
        elif backend == BackendType.GRAPH:
            return await self._retrieve_from_graph(query, plan, classification, context)
        
        else:
            self._logger.warning(f"Unknown backend: {backend.value}")
            return []
    
    async def _retrieve_from_vector(
        self,
        query: str,
        plan: RetrievalPlan,
        context: RequestContext,
    ) -> List[Evidence]:
        if self._vector_port is None or self._embedding_model is None:
            return []

        try:
            embedding = await self._get_embedding(query)
            return await vector_similarity_search(
                self._vector_port,
                self._collection,
                embedding,
                plan.top_k * 2,
                context.tenant_id,
                context.acl,
            )
        except Exception as e:
            self._logger.error("Vector retrieval failed: %s", e)
            return []
    
    async def _retrieve_from_sql(
        self,
        query: str,
        classification: ClassificationResult,
        context: RequestContext
    ) -> List[Evidence]:
        if self._sql_port is None:
            return []

        try:
            results: List[Evidence] = []
            seen: set = set()
            entities = classification.entities

            async def add_row(row: dict, score: float = 1.0) -> None:
                doc_id = str(row.get("doc_id", ""))
                if not doc_id or doc_id in seen:
                    return
                meta = row.get("metadata") or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except json.JSONDecodeError:
                        meta = {"raw": meta}
                if not check_acl({**meta, "doc_id": doc_id}, context.acl):
                    return
                seen.add(doc_id)
                results.append(
                    Evidence(
                        id=doc_id,
                        content=row.get("content", "") or "",
                        score=score,
                        source_type=SourceType.SQL,
                        citation=row.get("title"),
                        metadata={**meta, "doc_id": doc_id},
                    )
                )

            for entity in entities:
                if self._doc_store:
                    row = await self._doc_store.get_by_doc_id(entity, context.tenant_id)
                    if row:
                        await add_row(row)
                        continue
                rows_result = self._sql_port.select_many(
                    table="documents",
                    columns=["doc_id", "title", "content", "metadata"],
                    where={"doc_id": entity, "tenant_id": context.tenant_id},
                    limit=10,
                )
                rows = await rows_result if asyncio.iscoroutine(rows_result) else rows_result
                for row in rows:
                    await add_row(row)

            if not results and self._doc_store and classification.keywords:
                rows = await self._doc_store.search_by_keywords(
                    context.tenant_id,
                    classification.keywords,
                    limit=10,
                )
                for row in rows:
                    await add_row(row, score=0.8)

            return results

        except Exception as e:
            self._logger.error("SQL retrieval failed: %s", e)
            return []
    
    async def _retrieve_from_graph(
        self,
        query: str,
        plan: RetrievalPlan,
        classification: ClassificationResult,
        context: RequestContext
    ) -> List[Evidence]:
        if self._graph_port is None:
            return []
        
        try:
            entities = classification.entities
            if not entities:
                return []
            
            results = []
            hop = max(plan.graph_hop, 1)
            for entity in entities[:3]:
                paths = []
                if hasattr(self._graph_port, "k_hop_subgraph"):
                    paths = await asyncio.to_thread(
                        self._graph_port.k_hop_subgraph,
                        [entity],
                        hop,
                    )
                elif hasattr(self._graph_port, "get_k_hop_subgraph"):
                    subgraph = await self._graph_port.get_k_hop_subgraph(
                        entity_id=entity,
                        k=hop,
                        tenant_id=context.tenant_id,
                    )
                    path_text = self._linearize_subgraph(subgraph)
                    if path_text:
                        paths = [type("Path", (), {"text_representation": path_text})()]

                for path in paths or []:
                    path_text = getattr(path, "text_representation", None) or str(path)
                    if not path_text:
                        continue
                    results.append(
                        Evidence(
                            id=f"graph_{entity}",
                            content=path_text,
                            score=0.9,
                            source_type=SourceType.GRAPH,
                            metadata={"entity": entity, "hop": hop},
                        )
                    )
            
            return results
            
        except Exception as e:
            self._logger.error(f"Graph retrieval failed: {e}")
            return []
    
    async def _retrieve_from_cache(
        self,
        query: str,
        context: RequestContext
    ) -> List[Evidence]:
        if not self._enable_cache or self._cache_port is None:
            return []
        
        try:
            cache_key = self._get_cache_key(query, context.tenant_id)
            cached = await self._get_from_cache(cache_key)
            
            if cached and not cached.empty:
                return cached.evidences
            
            return []
            
        except Exception as e:
            self._logger.warning(f"Cache retrieval failed: {e}")
            return []
    
    def _linearize_subgraph(self, subgraph: Dict[str, Any]) -> str:
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])
        
        if not nodes:
            return ""
        
        lines = []
        for node in nodes[:10]:
            label = node.get("label", node.get("id", ""))
            props = node.get("properties", {})
            if props:
                prop_str = ", ".join(f"{k}={v}" for k, v in list(props.items())[:3])
                lines.append(f"{label} ({prop_str})")
            else:
                lines.append(label)
        
        for edge in edges[:10]:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            rel = edge.get("type", "related_to")
            lines.append(f"{src} --[{rel}]--> {tgt}")
        
        return "\n".join(lines)
    
    async def _get_embedding(self, text: str) -> List[float]:
        if self._embedding_model is None:
            raise RuntimeError("Embedding model not configured")
        
        if hasattr(self._embedding_model, 'aembed'):
            embeddings = await self._embedding_model.aembed([text])
            return embeddings[0]
        elif hasattr(self._embedding_model, 'embed'):
            embeddings = self._embedding_model.embed([text])
            return embeddings[0]
        else:
            raise RuntimeError("Embedding model has no embed method")
    
    def _get_cache_key(self, queries: Any, tenant_id: str) -> str:
        if isinstance(queries, list):
            payload = "|".join(sorted(queries))
        else:
            payload = str(queries)
        query_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return f"{tenant_id}:rag:qc:{query_hash}"
    
    async def _get_from_cache(self, key: str) -> Optional[EvidenceBundle]:
        if not self._enable_cache or self._cache_port is None:
            return None

        try:
            cached = await cache_get(self._cache_port, key)
            if cached:
                return bundle_from_cache_dict(cached)
        except Exception as e:
            self._logger.warning("Cache get failed: %s", e)

        return None

    async def _set_cache(self, key: str, bundle: EvidenceBundle) -> None:
        if not self._enable_cache or self._cache_port is None:
            return

        try:
            await cache_set(
                self._cache_port,
                key,
                bundle_to_cache_dict(bundle),
                self._cache_ttl,
            )
        except Exception as e:
            self._logger.warning("Cache set failed: %s", e)
