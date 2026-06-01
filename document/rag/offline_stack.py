"""兼容层：请使用 document.rag.bootstrap.offline。"""
from document.rag.bootstrap.offline import (  # noqa: F401
    build_offline_ingest_port,
    create_offline_index_service,
    load_offline_config,
    resolve_offline_embedding,
)

__all__ = [
    "build_offline_ingest_port",
    "create_offline_index_service",
    "load_offline_config",
    "resolve_offline_embedding",
]
