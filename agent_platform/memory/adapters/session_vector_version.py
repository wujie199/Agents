from __future__ import annotations

from typing import Any, Optional


def compute_session_vector_index_version(
    cfg: dict[str, Any],
    *,
    config_dir: Optional[str] = None,
    models: Any = None,
) -> str:
    """Derive a stable version string from embedding role (config/models.yml)."""
    explicit = cfg.get("session_vector_index_version")
    if explicit:
        return str(explicit)

    try:
        if models is None and config_dir:
            from document.rag.bootstrap.model_bridge import ensure_model_registry

            models = ensure_model_registry(None, config_dir=config_dir)
        if models is not None:
            return models.get_embedding_version_key("embedding")
    except Exception:
        pass

    return "embedding:unknown"
