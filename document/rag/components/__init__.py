"""RAG 组件子包 — 每个功能点一个子包，自包含 port + 实现 + registry。"""

from document.rag.components.embedding import build_embedding
from document.rag.components.rerank import build_rerank
from document.rag.components.ingest import build_ingest
from document.rag.components.metadata import build_metadata_enricher
from document.rag.components.storage import build_bm25_index

__all__ = [
    "build_embedding",
    "build_rerank",
    "build_ingest",
    "build_metadata_enricher",
    "build_bm25_index",
]
