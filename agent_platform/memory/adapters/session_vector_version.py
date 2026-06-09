from __future__ import annotations

from typing import Any, Optional


def compute_session_vector_index_version(
    cfg: dict[str, Any],
    *,
    config_dir: Optional[str] = None,
) -> str:
    """Derive a stable version string from embedding backend + model + dimension."""
    explicit = cfg.get("session_vector_index_version")
    if explicit:
        return str(explicit)

    backend = str(cfg.get("session_embedding_backend", "mock")).lower()
    dim = int(cfg.get("session_embedding_dim", 64))

    if backend == "local_bge" and config_dir:
        try:
            from document.rag.config import load_rag_pipeline_config

            rag_cfg = load_rag_pipeline_config(config_dir=config_dir)
            model = rag_cfg.embedding.model_path or "local_bge"
            return f"{backend}:{model}:{dim}"
        except Exception:
            pass

    return f"{backend}:{dim}"
