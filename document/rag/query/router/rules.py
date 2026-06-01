from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

from document.rag.query.router.classifier import QueryType, ClassificationResult


class BackendType(str, Enum):
    REDIS_CACHE = "redis_cache"
    SQL = "sql"
    VECTOR = "vector"
    GRAPH = "graph"


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
    """
    æ ¹æ® query_type åç¹å¾å³å®æ£ç´¢ç­ç¥ï¼
    1. è§åä¼åï¼ç¡®å®æ§ï¼å¯å®¡è®¡ï¼
    """
