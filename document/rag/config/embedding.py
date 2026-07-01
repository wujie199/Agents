from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding 适配器（backend 见 components/embedding/registry）。"""

    backend: str = "local_bge"
    model_path: Optional[str] = None
    device: Optional[str] = None
    normalize: bool = True
