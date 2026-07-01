"""Embedding 组件独立 registry — 按 backend 配置构建 EmbeddingPort。"""

from core.ports.rag.embedding import EmbeddingPort
from document.rag.config.pipeline import RagPipelineConfig


def build_embedding(cfg: RagPipelineConfig) -> EmbeddingPort:
    backend = (cfg.embedding.backend or "local_bge").lower()

    if backend == "mock":
        from document.rag.components.embedding.mock import MockEmbeddingModel
        return MockEmbeddingModel()

    if backend == "local_bge":
        from document.rag.components.embedding.local_bge import LocalBgeEmbedding
        emb = cfg.embedding
        return LocalBgeEmbedding(
            model_dir=emb.model_path,
            device=emb.device,
            normalize_embeddings=emb.normalize,
        )

    raise ValueError(f"未知 embedding backend: {backend!r}")


# 兼容旧名
resolve_local_embedding = build_embedding
