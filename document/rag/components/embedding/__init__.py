"""Embedding 组件 — port + 实现 + registry。"""

from document.rag.components.embedding.port import EmbeddingPort
from document.rag.components.embedding.registry import build_embedding

__all__ = ["EmbeddingPort", "build_embedding"]
