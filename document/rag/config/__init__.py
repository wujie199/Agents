"""RAG 配置子包 — re-export 所有配置类与加载函数，保持向后兼容。"""

from document.rag.config.ingest import IngestConfig
from document.rag.config.retrieval import RetrievalConfig
from document.rag.config.rewrite import RewriteConfig
from document.rag.config.embedding import EmbeddingConfig
from document.rag.config.rerank import RerankConfig
from document.rag.config.metadata import MetadataConfig
from document.rag.config.pipeline import (
    RagPipelineConfig,
    RAG_PIPELINE_PROFILES,
    compute_index_config_hash,
    detect_rag_profile_for_path,
    load_rag_pipeline_config,
    resolve_rag_pipeline_config_path,
)
from document.rag.config.rag_yaml import (
    RAG_PROFILES,
    resolve_rag_config_path,
    load_rag_eval_section,
    load_rag_scenarios_section,
)

__all__ = [
    "IngestConfig",
    "RetrievalConfig",
    "RewriteConfig",
    "EmbeddingConfig",
    "RerankConfig",
    "MetadataConfig",
    "RagPipelineConfig",
    "RAG_PROFILES",
    "RAG_PIPELINE_PROFILES",
    "compute_index_config_hash",
    "detect_rag_profile_for_path",
    "load_rag_pipeline_config",
    "resolve_rag_config_path",
    "resolve_rag_pipeline_config_path",
    "load_rag_eval_section",
    "load_rag_scenarios_section",
]
