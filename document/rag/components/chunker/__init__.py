"""Chunker 组件 — port + 实现 + registry。"""

from document.rag.components.chunker.port import ChunkerPort, Chunk, ChunkStrategy

__all__ = ["ChunkerPort", "Chunk", "ChunkStrategy"]
