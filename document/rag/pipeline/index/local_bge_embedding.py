from document.rag.adapters.embedding.local_bge import LocalBgeEmbedding
from document.rag.adapters.registry import build_embedding as resolve_local_embedding

__all__ = ["LocalBgeEmbedding", "resolve_local_embedding"]
