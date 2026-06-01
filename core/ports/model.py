from typing import Protocol, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    role: str
    profile: str
    provider: str
    degraded: bool = False
    circuit_open: bool = False
    fallback_index: int = 0


class ModelPort(Protocol):
    def get_model(self, role: str) -> Any:
        ...

    def get_model_info(self, role: str) -> ModelInfo:
        ...

    def invalidate_cache(self, role: Optional[str] = None) -> None:
        ...

    def get_embedding(self, role: str = "embedding") -> Any:
        ...

    def get_reranker(self, role: str = "rerank") -> Any:
        ...
