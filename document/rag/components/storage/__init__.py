"""Storage 组件 — BM25 等检索存储后端。"""

from document.rag.components.storage.registry import build_bm25_index

__all__ = ["build_bm25_index"]
