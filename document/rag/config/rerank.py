from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RerankConfig:
    """Rerank 适配器（backend: local_bge | mock | none）。"""

    backend: str = "local_bge"
    model_path: Optional[str] = None
    device: Optional[str] = None
