from pathlib import Path
from typing import Any, Optional

from agent_platform.memory.adapters.config_loader import load_memory_config


def build_session_vector_index(
    cfg: dict[str, Any],
    *,
    data_dir: str,
    config_dir: str,
) -> Optional[Any]:
    if not cfg.get("enable_session_vector_index", False):
        return None

    from agent_platform.memory.adapters.session_message_vector_index import (
        SessionMessageVectorIndex,
    )
    from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter

    backend = str(cfg.get("session_embedding_backend", "mock")).lower()
    if backend == "mock":
        from document.rag.adapters.embedding.mock import MockEmbeddingModel

        embedding = MockEmbeddingModel(
            dimension=int(cfg.get("session_embedding_dim", 64))
        )
    elif backend == "local_bge":
        from document.rag.adapters.registry import build_embedding
        from document.rag.config import load_rag_pipeline_config

        rag_cfg = load_rag_pipeline_config(config_dir=config_dir)
        embedding = build_embedding(rag_cfg, config_dir=config_dir)
    else:
        raise ValueError(f"未知 session_embedding_backend: {backend!r}")

    vector_dir = cfg.get("session_vector_dir", f"{data_dir}/session_vectors")
    Path(vector_dir).mkdir(parents=True, exist_ok=True)
    vector_port = ChromaVectorAdapter(persist_directory=str(vector_dir))
    collection = cfg.get("session_vector_collection", "session_messages")
    return SessionMessageVectorIndex(vector_port, embedding, collection=collection)
