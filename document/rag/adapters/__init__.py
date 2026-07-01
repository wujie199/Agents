"""向后兼容层 — 适配器已迁移至 components/ 子包。"""

from document.rag.components.embedding.registry import build_embedding
from document.rag.components.ingest.registry import build_ingest
from document.rag.components.metadata.registry import build_metadata_enricher
from document.rag.components.rerank.registry import build_rerank

__all__ = [
    "build_embedding",
    "build_ingest",
    "build_metadata_enricher",
    "build_rerank",
]
