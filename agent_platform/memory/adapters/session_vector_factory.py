from pathlib import Path
from typing import Any, Optional

from agent_platform.memory.adapters.config_loader import load_memory_config
from agent_platform.memory.adapters.session_vector_version import (
    compute_session_vector_index_version,
)


def build_session_vector_index(
    cfg: dict[str, Any],
    *,
    data_dir: str,
    config_dir: str,
    models: Any = None,
) -> Optional[Any]:
    if not cfg.get("enable_session_vector_index", False):
        return None

    from agent_platform.memory.adapters.session_message_vector_index import (
        SessionMessageVectorIndex,
    )
    from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter
    from document.rag.bootstrap.model_bridge import ensure_model_registry

    registry = ensure_model_registry(models, config_dir=config_dir)
    embedding = registry.get_embedding_port("embedding")

    vector_dir = cfg.get("session_vector_dir", f"{data_dir}/session_vectors")
    Path(vector_dir).mkdir(parents=True, exist_ok=True)
    vector_port = ChromaVectorAdapter(persist_directory=str(vector_dir))
    collection = cfg.get("session_vector_collection", "session_messages")
    batch_size = int(cfg.get("session_vector_embed_batch_size", 32))
    index_version = compute_session_vector_index_version(
        cfg, config_dir=config_dir, models=registry
    )
    return SessionMessageVectorIndex(
        vector_port,
        embedding,
        collection=collection,
        embed_batch_size=batch_size,
        index_version=index_version,
    )
