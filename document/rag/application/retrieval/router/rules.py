from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

from document.rag.application.retrieval.router.classifier import (
    QueryType,
    ClassificationResult,
)

_logger = logging.getLogger("rag.routing_rules")


class BackendType(str, Enum):
    REDIS_CACHE = "redis_cache"
    SQL = "sql"
    VECTOR = "vector"
    GRAPH = "graph"
    BM25 = "bm25"


class ExecutionOrder(str, Enum):
    PARALLEL = "parallel"
    CASCADE = "cascade"


@dataclass
class RetrievalPlan:
    primary: BackendType
    secondary: List[BackendType] = field(default_factory=list)
    order: ExecutionOrder = ExecutionOrder.PARALLEL
    fusion: str = "rrf"
    cache_policy: str = "read_through"
    graph_hop: int = 0
    top_k: int = 10
    rerank_top_n: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary": self.primary.value,
            "secondary": [b.value for b in self.secondary],
            "order": self.order.value,
            "fusion": self.fusion,
            "cache_policy": self.cache_policy,
            "graph_hop": self.graph_hop,
            "top_k": self.top_k,
            "rerank_top_n": self.rerank_top_n,
        }


class RoutingRules:
    """根据 query_type 与启用的后端生成 retrieval_plan。"""

    def __init__(
        self,
        enable_graph: bool = False,
        enable_sql: bool = False,
        enable_bm25: bool = False,
        *,
        default_top_k: int = 10,
        default_rerank_n: int = 5,
    ):
        self._enable_graph = enable_graph
        self._enable_sql = enable_sql
        self._enable_bm25 = enable_bm25
        self._default_top_k = default_top_k
        self._default_rerank_n = default_rerank_n

    def get_supported_backends(self) -> List[BackendType]:
        backends = [BackendType.REDIS_CACHE, BackendType.VECTOR]
        if self._enable_sql:
            backends.append(BackendType.SQL)
        if self._enable_graph:
            backends.append(BackendType.GRAPH)
        if self._enable_bm25:
            backends.append(BackendType.BM25)
        return backends

    def route(
        self,
        classification: ClassificationResult,
        context: Optional[Dict[str, Any]] = None,
    ) -> RetrievalPlan:
        qt = classification.query_type
        top_k = self._default_top_k
        rerank_n = self._default_rerank_n

        if qt == QueryType.FACTUAL_EXACT:
            secondary: List[BackendType] = []
            if self._enable_sql:
                secondary.append(BackendType.VECTOR)
            return RetrievalPlan(
                primary=BackendType.SQL,
                secondary=secondary,
                fusion="first_match",
                top_k=top_k,
                rerank_top_n=rerank_n,
            )

        if qt == QueryType.GRAPH:
            secondary = [BackendType.VECTOR] if self._enable_graph else []
            return RetrievalPlan(
                primary=BackendType.GRAPH,
                secondary=secondary,
                fusion="rrf",
                graph_hop=2,
                top_k=top_k,
                rerank_top_n=rerank_n,
            )

        if qt == QueryType.RELATIONAL:
            if self._enable_sql:
                return RetrievalPlan(
                    primary=BackendType.SQL,
                    secondary=[BackendType.VECTOR],
                    fusion="rrf",
                    top_k=top_k,
                    rerank_top_n=rerank_n,
                )

        if qt == QueryType.HYBRID:
            secondary = []
            if self._enable_sql:
                secondary.append(BackendType.SQL)
            if self._enable_graph:
                secondary.append(BackendType.GRAPH)
            if self._enable_bm25:
                secondary.append(BackendType.BM25)
            return RetrievalPlan(
                primary=BackendType.VECTOR,
                secondary=secondary,
                fusion="rrf",
                top_k=top_k,
                rerank_top_n=rerank_n,
            )

        # SEMANTIC_DOC / OPERATIONAL / default
        secondary = [BackendType.SQL] if self._enable_sql else []
        if self._enable_bm25:
            secondary.append(BackendType.BM25)
        return RetrievalPlan(
            primary=BackendType.VECTOR,
            secondary=secondary,
            fusion="rrf",
            top_k=top_k,
            rerank_top_n=rerank_n,
        )
