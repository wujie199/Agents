from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class RetrievalConfig:
    primary_backend: str = "vector"
    enable_rerank: bool = False
    enable_router: bool = True
    auto_route: bool = True
    enable_graph: bool = False
    enable_sql: bool = False
    use_mock_rerank_fallback: bool = True
    enable_hybrid: bool = False
    enable_vector_search: bool = True
    enable_bm25_search: bool = True
    vector_top_k: int = 10
    bm25_top_k: int = 10
    hybrid_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])
    fusion_strategy: str = "weighted"
    fusion_top_n: int = 10
    rerank_top_n: int = 5
    rerank_min_score: Optional[float] = 0.8
