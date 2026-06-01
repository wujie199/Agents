"""兼容层：在线 RAG 组装已迁至 document.rag.bootstrap.online。"""

from document.rag.bootstrap.online import (
    build_rag_stack,
    build_retrieval_router,
    resolve_embedding_model,
    resolve_rerank_model,
)

__all__ = [
    "build_rag_stack",
    "build_retrieval_router",
    "resolve_embedding_model",
    "resolve_rerank_model",
]
