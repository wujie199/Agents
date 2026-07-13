"""离线建库专用：仅组装 ingest / index / Chroma，不拉起 Agent 全栈。"""

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

_log = logging.getLogger("document.rag.bootstrap.offline")

from core.ports.index import IndexProfile
from document.rag.config import RagPipelineConfig, load_rag_pipeline_config
from document.rag.components.storage.registry import build_bm25_index
from document.rag.application.embedding.collection import effective_collection_name
from document.rag.application.indexing.service import IndexService
from document.rag.application.ingest_factory import build_ingest_pipeline
from document.rag.bootstrap.model_bridge import apply_models_to_rag_config
from document.rag.bootstrap.online import resolve_embedding_model


def resolve_offline_embedding(
    cfg: RagPipelineConfig,
    models: Any = None,
) -> Any:
    """离线建库 embedding（见 config/models.yml roles.embedding）。"""
    return resolve_embedding_model(models, cfg)


def create_offline_embedding_cache(
    data_dir: Path,
    cfg: RagPipelineConfig,
) -> Any | None:
    """离线 embedding 文件缓存（enable_embedding_cache_read 时启用）。"""
    if not cfg.embedding.enable_embedding_cache_read:
        return None
    from document.rag.components.storage.embedding_file_cache import (
        EmbeddingFileCacheAdapter,
    )

    cache_dir = data_dir / "embedding_cache"
    _log.info("离线 embedding 缓存目录: %s", cache_dir)
    return EmbeddingFileCacheAdapter(cache_dir)


def create_offline_index_service(
    data_dir: Path,
    cfg: RagPipelineConfig,
    *,
    config_dir: str = "config",
    index_profile: IndexProfile = IndexProfile.VECTOR_ONLY,
    enable_bm25: bool = True,
    models: Any = None,
) -> Tuple[IndexService, str]:
    """
    创建 IndexService 与 Chroma 持久化路径。

    Returns:
        (index_service, chroma_persist_dir)
    """
    from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter

    chroma_dir = str(data_dir / "chroma_dev")
    vector_port = ChromaVectorAdapter(persist_directory=chroma_dir)
    embedding = resolve_offline_embedding(cfg, models=models)

    sql_port = None
    graph_port = None
    if index_profile in (IndexProfile.SQL_SIDECAR, IndexProfile.FULL) and cfg.retrieval.enable_sql:
        from agent_platform.storage.adapters.sqlite.relational_adapter import (
            AsyncSQLiteRelationalAdapter,
        )

        sql_port = AsyncSQLiteRelationalAdapter(
            db_path=str(data_dir / "dev_archive.db"),
            pool_size=3,
            timeout=10.0,
        )
    if index_profile in (IndexProfile.GRAPH_SIDECAR, IndexProfile.FULL) and cfg.enable_graph_index:
        from agent_platform.storage.adapters.graph.memory_graph_adapter import (
            MemoryGraphAdapter,
        )

        graph_port = MemoryGraphAdapter()

    bm25_index = None
    if enable_bm25:
        bm25_index = build_bm25_index(data_dir, cfg)

    cache_port = create_offline_embedding_cache(data_dir, cfg)

    index_service = IndexService(
        vector_port=vector_port,
        embedding_model=embedding,
        config=cfg,
        cache_port=cache_port,
        sql_port=sql_port,
        graph_port=graph_port,
        bm25_index=bm25_index,
    )
    _log.info(
        "IndexService collection=%s chroma=%s",
        effective_collection_name(cfg),
        chroma_dir,
    )
    return index_service, chroma_dir


def load_offline_config(
    config_dir: str,
    models: Any = None,
    *,
    config_path: Optional[str] = None,
    profile: Optional[str] = None,
) -> RagPipelineConfig:
    from document.rag.config.pipeline import resolve_rag_pipeline_config_path

    resolved = config_path or resolve_rag_pipeline_config_path(
        config_dir=config_dir,
        profile=profile,
    )
    cfg = load_rag_pipeline_config(config_path=resolved, config_dir=config_dir)
    cfg, _ = apply_models_to_rag_config(cfg, models, config_dir=config_dir)
    return cfg


def build_offline_ingest_port(cfg: RagPipelineConfig):
    return build_ingest_pipeline(cfg)
