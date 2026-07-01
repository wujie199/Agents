"""向后兼容层 — 所有配置类已迁移至 config/ 子包。"""

from document.rag.config.ingest import IngestConfig
from document.rag.config.retrieval import RetrievalConfig
from document.rag.config.rewrite import RewriteConfig
from document.rag.config.embedding import EmbeddingConfig
from document.rag.config.rerank import RerankConfig
from document.rag.config.metadata import MetadataConfig
from document.rag.config.pipeline import (
    RagPipelineConfig,
    RAG_PIPELINE_PROFILES,
    warn_local_model_paths,
    compute_index_config_hash,
    detect_rag_profile_for_path,
    load_rag_pipeline_config,
    resolve_rag_pipeline_config_path,
)

__all__ = [
    "IngestConfig",
    "RetrievalConfig",
    "RewriteConfig",
    "EmbeddingConfig",
    "RerankConfig",
    "MetadataConfig",
    "RagPipelineConfig",
    "RAG_PIPELINE_PROFILES",
    "warn_local_model_paths",
    "compute_index_config_hash",
    "detect_rag_profile_for_path",
    "load_rag_pipeline_config",
    "resolve_rag_pipeline_config_path",
]
