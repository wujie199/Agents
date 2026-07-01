"""向后兼容层 — 所有 registry 函数已迁移至 components/ 子包。"""

from document.rag.components.embedding.registry import build_embedding, resolve_local_embedding
from document.rag.components.rerank.registry import build_rerank
from document.rag.components.ingest.registry import build_ingest
from document.rag.components.metadata.registry import build_metadata_enricher
from document.rag.components.storage.registry import build_bm25_index

__all__ = [
    "build_embedding",
    "resolve_local_embedding",
    "build_rerank",
    "build_ingest",
    "build_metadata_enricher",
    "build_bm25_index",
]
