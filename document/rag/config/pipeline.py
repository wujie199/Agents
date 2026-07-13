from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from document.model_mount import warn_if_unmounted
from document.rag.config.embedding import EmbeddingConfig
from document.rag.config.rerank import RerankConfig
from document.rag.config.metadata import MetadataConfig
from document.rag.config.ingest import IngestConfig
from document.rag.config.retrieval import RetrievalConfig
from document.rag.config.rewrite import (
    RewriteConfig,
    TwoStageConfig,
    parse_rewrite_profiles,
)
from document.rag.config.chunk_pipeline import ChunkPipelineConfig, parse_chunk_pipeline_config
from document.rag.config.rag_yaml import (
    RAG_PROFILES,
    RAG_PIPELINE_PROFILES,
    load_rag_yaml_document,
    resolve_rag_config_path,
    resolve_rag_pipeline_config_path,
)


@dataclass(frozen=True)
class RagPipelineConfig:
    collection_name: str = "agent"
    enable_vector_index: bool = True
    enable_graph_index: bool = False
    chunk_size: int = 200
    chunk_overlap: int = 20
    chunk_strategy: str = "seven_step"
    default_top_k: int = 10
    rerank_top_n: int = 5
    enable_cache: bool = True
    cache_ttl_seconds: int = 900
    model_version: str = "bge-small-zh-v1.5-local"
    embedding_batch_size: int = 32
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rewrite: RewriteConfig = field(default_factory=RewriteConfig)
    enable_chunk_dedupe: bool = False
    enable_semantic_dedupe: bool = False
    semantic_dedupe_threshold: float = 0.85
    cleaners: Optional[Dict[str, Any]] = None
    chunk_pipeline: ChunkPipelineConfig = field(default_factory=ChunkPipelineConfig)


def warn_model_instances(config_dir: str = "config") -> None:
    """外置盘未挂载时打印提醒（模型路径来自 config/models.yml）。"""
    try:
        from agent_platform.model.registry import ModelRegistry

        reg = ModelRegistry(config_path=str(Path(config_dir) / "models.yml"))
        emb = reg._get_profile_for_role("embedding")
        if emb.provider == "local_bge" and emb.model_path:
            warn_if_unmounted(
                emb.model_path,
                purpose="RAG/L2 embedding",
                env_hint="RAG_EMBEDDING_MODEL_PATH",
            )
        rerank = reg._get_profile_for_role("rerank")
        if rerank.provider == "local_bge" and rerank.model_path:
            warn_if_unmounted(
                rerank.model_path,
                purpose="RAG rerank",
                env_hint="RAG_RERANK_MODEL_PATH",
            )
        ocr_root = reg.get_ocr_model_root()
        if ocr_root:
            warn_if_unmounted(
                ocr_root,
                purpose="OCR 文档摄取",
                env_hint="OCR_MODEL_ROOT",
            )
    except Exception:
        pass


def compute_index_config_hash(cfg: RagPipelineConfig) -> str:
    """索引相关配置指纹，用于 manifest 跳过/失效判断。"""
    payload = {
        "model_version": cfg.model_version,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "chunk_strategy": cfg.chunk_strategy,
        "collection_name": cfg.collection_name,
        "enable_chunk_dedupe": cfg.enable_chunk_dedupe,
        "enable_semantic_dedupe": cfg.enable_semantic_dedupe,
        "chunk_pipeline": asdict(cfg.chunk_pipeline),
        "embedding": asdict(cfg.embedding),
        "cleaners": cfg.cleaners,
        "ingest": asdict(cfg.ingest),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return sha256(raw.encode("utf-8")).hexdigest()[:16]


def _apply_rag_env_overrides(cfg: RagPipelineConfig) -> RagPipelineConfig:
    """环境变量覆盖检索行为（模型路径见 config/models.yml + ModelRegistry）。"""
    retrieval_updates: dict = {}
    if os.environ.get("RAG_USE_MOCK_RERANK_FALLBACK", "").lower() in (
        "0",
        "false",
        "no",
    ):
        retrieval_updates["use_mock_rerank_fallback"] = False
    if os.environ.get("RAG_ENABLE_ROUTER", "").lower() in ("1", "true", "yes"):
        retrieval_updates["enable_router"] = True
    if not retrieval_updates:
        return cfg
    retrieval = replace(cfg.retrieval, **retrieval_updates)
    return replace(cfg, retrieval=retrieval)


RAG_PIPELINE_PROFILES = RAG_PROFILES


def resolve_rag_pipeline_config_path(
    config_dir: str = "config",
    profile: Optional[str] = None,
) -> str:
    """解析 RAG YAML 路径（兼容旧 API）。"""
    return resolve_rag_config_path(config_dir, profile)


def detect_rag_profile_for_path(file_path: str | Path) -> str:
    """按扩展名推断建库 profile（Web 上传等）。"""
    ext = Path(file_path).suffix.lower().lstrip(".")
    if ext in ("docx", "doc"):
        return "contract"
    if ext == "pdf":
        return "faq"
    return "faq"


def load_rag_pipeline_config(
    config_path: Optional[str] = None,
    config_dir: str = "config",
) -> RagPipelineConfig:
    if config_path is None:
        config_path = resolve_rag_config_path(config_dir)

    path = Path(config_path)
    if not path.exists():
        return RagPipelineConfig()

    raw = load_rag_yaml_document(path, config_dir=config_dir)

    ingest_raw = raw.get("ingest") or {}
    retrieval_raw = raw.get("retrieval") or {}
    rewrite_raw = raw.get("rewrite") or {}
    embedding_raw = raw.get("embedding") or {}
    rerank_raw = raw.get("rerank") or {}
    metadata_raw = raw.get("metadata") or {}

    collection_name = raw.get("collection_name", "agent")
    chroma: dict = dict(raw.get("storage", {}).get("chroma") or {})
    legacy_chroma = dict(raw.get("storage", {}).get("legacy") or {})
    for key, value in legacy_chroma.items():
        chroma.setdefault(key, value)

    cfg = RagPipelineConfig(
        collection_name=collection_name,
        enable_vector_index=raw.get("enable_vector_index", True),
        enable_graph_index=raw.get("enable_graph_index", False),
        chunk_size=raw.get("chunk_size", 200),
        chunk_overlap=raw.get("chunk_overlap", 20),
        chunk_strategy=str(raw.get("chunk_strategy", "seven_step")),
        default_top_k=raw.get("default_top_k", 10),
        rerank_top_n=raw.get("rerank_top_n", 5),
        enable_cache=raw.get("enable_cache", True),
        cache_ttl_seconds=raw.get("cache_ttl_seconds", 900),
        model_version=raw.get("model_version", "bge-small-zh-v1.5-local"),
        embedding_batch_size=raw.get("embedding_batch_size", 32),
        embedding=EmbeddingConfig(
            backend=str(embedding_raw.get("backend", "local_bge")),
            model_path=embedding_raw.get("model_path"),
            device=embedding_raw.get("device"),
            normalize=embedding_raw.get("normalize", True),
            max_tokens=int(embedding_raw.get("max_tokens", 512)),
            truncate_marker=str(embedding_raw.get("truncate_marker", "[...]")),
            query_instruction=str(
                embedding_raw.get(
                    "query_instruction",
                    "为这个句子生成表示以用于检索相关文章：",
                )
            ),
            doc_instruction=str(embedding_raw.get("doc_instruction", "")),
            batch_size=int(
                embedding_raw.get("batch_size", raw.get("embedding_batch_size", 32))
            ),
            batch_size_min=int(embedding_raw.get("batch_size_min", 1)),
            oom_halve_retry=bool(embedding_raw.get("oom_halve_retry", True)),
            force_l2_normalize=bool(embedding_raw.get("force_l2_normalize", True)),
            verify_unit_norm=bool(embedding_raw.get("verify_unit_norm", True)),
            unit_norm_tolerance=float(embedding_raw.get("unit_norm_tolerance", 0.02)),
            reject_zero_vectors=bool(embedding_raw.get("reject_zero_vectors", True)),
            matryoshka_dim=embedding_raw.get("matryoshka_dim"),
            write_max_retries=int(embedding_raw.get("write_max_retries", 3)),
            dlq_path=str(embedding_raw.get("dlq_path", "data/rag_offline/embedding_dlq.jsonl")),
            enable_chunk_incremental=bool(
                embedding_raw.get("enable_chunk_incremental", True)
            ),
            enable_embedding_cache_read=bool(
                embedding_raw.get("enable_embedding_cache_read", True)
            ),
            incremental_on_reindex=bool(
                embedding_raw.get("incremental_on_reindex", True)
            ),
            force_full_delete_on_reindex=bool(
                embedding_raw.get("force_full_delete_on_reindex", False)
            ),
            versioned_collection=bool(embedding_raw.get("versioned_collection", False)),
        ),
        rerank=RerankConfig(
            backend=str(rerank_raw.get("backend", "local_bge")),
            model_path=rerank_raw.get("model_path"),
            device=rerank_raw.get("device"),
        ),
        metadata=MetadataConfig(
            enabled=bool(metadata_raw.get("enabled", True)),
            backend=str(metadata_raw.get("backend", "rule_keyword")),
            rules_path=metadata_raw.get("rules_path"),
            max_tags=int(metadata_raw.get("max_tags", 32)),
            tag_filename=bool(metadata_raw.get("tag_filename", True)),
            rules=metadata_raw.get("rules"),
            extension_tags=metadata_raw.get("extension_tags"),
        ),
        ingest=IngestConfig(
            routing=ingest_raw.get("routing", "simplified"),
            mode=ingest_raw.get("mode", "ocr_only"),
            plain_text_formats=ingest_raw.get("plain_text_formats", ["txt", "md"]),
            ocr_backend=ingest_raw.get("ocr_backend", "auto"),
            language=ingest_raw.get("language", "ch"),
            word_to_pdf=ingest_raw.get("word_to_pdf", True),
            word_converter=ingest_raw.get("word_converter", "libreoffice"),
            pdf_dpi=int(ingest_raw.get("pdf_dpi", 200)),
            ocr_use_layout=ingest_raw.get("ocr_use_layout", True),
            enable_cleaning=ingest_raw.get("enable_cleaning", True),
            ocr_postprocess=ingest_raw.get("ocr_postprocess", True),
            ocr_preserve_structure=bool(ingest_raw.get("ocr_preserve_structure", True)),
            cleaning_level=str(ingest_raw.get("cleaning_level", "standard")),
            ocr_model_root=ingest_raw.get("ocr_model_root"),
            ocr_device=str(ingest_raw.get("ocr_device", "cpu")),
            ocr_preprocess=str(ingest_raw.get("ocr_preprocess", "auto")),
            ocr_enable_formula=bool(ingest_raw.get("ocr_enable_formula", True)),
            ocr_formula_model=ingest_raw.get("ocr_formula_model"),
            ocr_max_attempts=int(ingest_raw.get("ocr_max_attempts", 3)),
            ocr_layout_threshold=float(
                ingest_raw.get("ocr_layout_threshold", 0.5)
            ),
            ocr_layout_score_threshold=float(
                ingest_raw.get("ocr_layout_score_threshold", 0.5)
            ),
            ocr_fast=bool(ingest_raw.get("ocr_fast", False)),
            ocr_table_e2e=bool(ingest_raw.get("ocr_table_e2e", False)),
            ocr_enable_mkldnn=bool(ingest_raw.get("ocr_enable_mkldnn", True)),
            enable_header_footer_dedup=bool(
                ingest_raw.get("enable_header_footer_dedup", False)
            ),
            header_footer_threshold=float(
                ingest_raw.get("header_footer_threshold", 0.3)
            ),
            enable_pdf_routing=bool(ingest_raw.get("enable_pdf_routing", True)),
            pdf_threads=max(1, int(ingest_raw.get("pdf_threads", 1))),
        ),
        enable_chunk_dedupe=bool(raw.get("enable_chunk_dedupe", False)),
        enable_semantic_dedupe=bool(raw.get("enable_semantic_dedupe", False)),
        semantic_dedupe_threshold=float(raw.get("semantic_dedupe_threshold", 0.85)),
        cleaners=raw.get("cleaners"),
        chunk_pipeline=parse_chunk_pipeline_config(raw.get("chunk_pipeline")),
        retrieval=RetrievalConfig(
            primary_backend=retrieval_raw.get("primary_backend", "vector"),
            enable_rerank=retrieval_raw.get("enable_rerank", False),
            enable_router=retrieval_raw.get("enable_router", True),
            auto_route=retrieval_raw.get("auto_route", True),
            enable_graph=retrieval_raw.get("enable_graph", False),
            enable_sql=retrieval_raw.get("enable_sql", False),
            use_mock_rerank_fallback=retrieval_raw.get(
                "use_mock_rerank_fallback", True
            ),
            enable_hybrid=retrieval_raw.get(
                "enable_hybrid", chroma.get("hybrid_search", False)
            ),
            enable_vector_search=retrieval_raw.get("enable_vector_search", True),
            enable_bm25_search=retrieval_raw.get("enable_bm25_search", True),
            vector_top_k=int(
                retrieval_raw.get("vector_top_k", chroma.get("k", raw.get("default_top_k", 10)))
            ),
            bm25_top_k=int(
                retrieval_raw.get("bm25_top_k", chroma.get("bm25_k", raw.get("default_top_k", 10)))
            ),
            hybrid_weights=list(
                retrieval_raw.get("hybrid_weights")
                or chroma.get("hybrid_weights")
                or [0.5, 0.5]
            ),
            fusion_strategy=str(retrieval_raw.get("fusion_strategy", "weighted")),
            fusion_top_n=int(
                retrieval_raw.get("fusion_top_n", raw.get("default_top_k", 10))
            ),
            rerank_top_n=int(
                retrieval_raw.get("rerank_top_n", raw.get("rerank_top_n", 5))
            ),
            rerank_min_score=retrieval_raw.get("rerank_min_score", 0.8),
        ),
        rewrite=RewriteConfig(
            enable_hyde=rewrite_raw.get("enable_hyde", False),
            enable_multi_query=rewrite_raw.get("enable_multi_query", True),
            multi_query_count=int(rewrite_raw.get("multi_query_count", 3)),
            enable_rule_rewrite=rewrite_raw.get("enable_rule_rewrite", True),
            rule_max_queries=int(rewrite_raw.get("rule_max_queries", 4)),
            maintenance_source_boost=float(
                rewrite_raw.get("maintenance_source_boost", 0.12)
            ),
            maintenance_post_rerank_boost=float(
                rewrite_raw.get("maintenance_post_rerank_boost", 0.18)
            ),
            faq_non_maintenance_penalty=float(
                rewrite_raw.get("faq_non_maintenance_penalty", 0.12)
            ),
            llm_rewrite_once=rewrite_raw.get("llm_rewrite_once", True),
            max_hybrid_queries=int(rewrite_raw.get("max_hybrid_queries", 6)),
            hybrid_search_concurrency=int(
                rewrite_raw.get("hybrid_search_concurrency", 4)
            ),
            default_profile=str(
                rewrite_raw.get("default_profile", "generic_knowledge")
            ),
            profiles=parse_rewrite_profiles(rewrite_raw.get("profiles")),
            two_stage=TwoStageConfig(
                enabled=bool(
                    (rewrite_raw.get("two_stage") or {}).get("enabled", True)
                ),
                top1_min_rerank=float(
                    (rewrite_raw.get("two_stage") or {}).get(
                        "top1_min_rerank", 0.75
                    )
                ),
                require_maintenance_source=bool(
                    (rewrite_raw.get("two_stage") or {}).get(
                        "require_maintenance_source", True
                    )
                ),
            ),
        ),
    )
    cfg = _apply_rag_env_overrides(cfg)
    warn_model_instances(config_dir)
    return cfg
