"""Rerank 组件独立 registry — 按 backend 配置构建 RerankPort。"""

from typing import Optional

from core.ports.rag.rerank import RerankPort
from document.rag.config.pipeline import RagPipelineConfig


def build_rerank(cfg: RagPipelineConfig) -> Optional[RerankPort]:
    if not cfg.retrieval.enable_rerank:
        return None

    rerank_raw = getattr(cfg, "rerank", None)
    backend = "local_bge"
    if rerank_raw is not None:
        backend = (rerank_raw.backend or "local_bge").lower()
    backend = backend.lower()

    if backend == "none":
        return None

    if backend == "mock":
        from document.rag.components.rerank.mock import MockRerankModel
        return MockRerankModel()

    if backend == "local_bge":
        from document.rag.components.rerank.local_bge import LocalBgeReranker
        r = rerank_raw
        return LocalBgeReranker(
            model_dir=getattr(r, "model_path", None) if r else None,
            device=getattr(r, "device", None) if r else None,
        )

    if cfg.retrieval.use_mock_rerank_fallback:
        from document.rag.components.rerank.mock import MockRerankModel
        return MockRerankModel()

    return None
