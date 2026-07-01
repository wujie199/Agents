from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from document.model_mount import warn_if_unmounted
from document.rag.config.embedding import EmbeddingConfig
from document.rag.config.rerank import RerankConfig
from document.rag.config.metadata import MetadataConfig
from document.rag.config.ingest import IngestConfig
from document.rag.config.retrieval import RetrievalConfig
from document.rag.config.rewrite import RewriteConfig


@dataclass(frozen=True)
class RagPipelineConfig:
    collection_name: str = "agent"
    enable_vector_index: bool = True
    enable_graph_index: bool = False
    chunk_size: int = 200
    chunk_overlap: int = 20
    chunk_strategy: str = "faq"
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


def warn_local_model_paths(cfg: RagPipelineConfig) -> None:
    """兼容旧调用；模型路径警告见 warn_model_instances。"""
    _ = cfg


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
        "embedding": asdict(cfg.embedding),
        "cleaners": cfg.cleaners,
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


RAG_PIPELINE_PROFILES = frozenset({"faq", "contract"})


def resolve_rag_pipeline_config_path(
    config_dir: str = "config",
    profile: Optional[str] = None,
) -> str:
    """解析 RAG 管道 YAML 路径：profile > RAG_PIPELINE_CONFIG 环境变量 > 默认。"""
    if profile:
        key = profile.lower().strip()
        if key not in RAG_PIPELINE_PROFILES:
            raise ValueError(
                f"未知 profile: {profile!r}，可选: {', '.join(sorted(RAG_PIPELINE_PROFILES))}"
            )
        path = Path(config_dir) / f"rag_pipeline.{key}.yml"
        if not path.exists():
            raise FileNotFoundError(f"profile 配置文件不存在: {path}")
        return str(path)

    env_path = os.environ.get("RAG_PIPELINE_CONFIG")
    if env_path:
        return env_path
    return str(Path(config_dir) / "rag_pipeline.yml")


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
        env_path = os.environ.get("RAG_PIPELINE_CONFIG")
        if env_path:
            config_path = env_path
        else:
            config_path = str(Path(config_dir) / "rag_pipeline.yml")

    path = Path(config_path)
    if not path.exists():
        return RagPipelineConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    ingest_raw = raw.get("ingest") or {}
    retrieval_raw = raw.get("retrieval") or {}
    rewrite_raw = raw.get("rewrite") or {}
    embedding_raw = raw.get("embedding") or {}
    rerank_raw = raw.get("rerank") or {}
    metadata_raw = raw.get("metadata") or {}

    chroma_path = Path(config_dir) / "chroma.yml"
    default_rules = str(Path(config_dir) / "metadata_tagging.yml")
    collection_name = raw.get("collection_name", "agent")
    chroma: dict = dict(raw.get("storage", {}).get("chroma") or {})
    if chroma_path.exists():
        with open(chroma_path, "r", encoding="utf-8") as f:
            legacy_chroma = yaml.safe_load(f) or {}
        for key, value in legacy_chroma.items():
            chroma.setdefault(key, value)
    if path.name == "rag_pipeline.yml" and not raw.get("collection_name"):
        collection_name = chroma.get("collection_name", collection_name)

    cfg = RagPipelineConfig(
        collection_name=collection_name,
        enable_vector_index=raw.get("enable_vector_index", True),
        enable_graph_index=raw.get("enable_graph_index", False),
        chunk_size=raw.get("chunk_size", 200),
        chunk_overlap=raw.get("chunk_overlap", 20),
        chunk_strategy=str(raw.get("chunk_strategy", "faq")),
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
        ),
        rerank=RerankConfig(
            backend=str(rerank_raw.get("backend", "local_bge")),
            model_path=rerank_raw.get("model_path"),
            device=rerank_raw.get("device"),
        ),
        metadata=MetadataConfig(
            enabled=bool(metadata_raw.get("enabled", True)),
            backend=str(metadata_raw.get("backend", "rule_keyword")),
            rules_path=metadata_raw.get("rules_path") or default_rules,
            max_tags=int(metadata_raw.get("max_tags", 32)),
            tag_filename=bool(metadata_raw.get("tag_filename", True)),
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
            cleaning_level=str(ingest_raw.get("cleaning_level", "standard")),
            ocr_model_root=ingest_raw.get("ocr_model_root"),
            ocr_device=str(ingest_raw.get("ocr_device", "cpu")),
            ocr_preprocess=str(ingest_raw.get("ocr_preprocess", "auto")),
            ocr_enable_formula=bool(ingest_raw.get("ocr_enable_formula", True)),
            ocr_formula_model=ingest_raw.get("ocr_formula_model"),
            ocr_max_attempts=int(ingest_raw.get("ocr_max_attempts", 3)),
            ocr_fast=bool(ingest_raw.get("ocr_fast", True)),
            ocr_table_e2e=bool(ingest_raw.get("ocr_table_e2e", False)),
            ocr_enable_mkldnn=bool(ingest_raw.get("ocr_enable_mkldnn", True)),
            enable_header_footer_dedup=bool(
                ingest_raw.get("enable_header_footer_dedup", False)
            ),
            header_footer_threshold=float(
                ingest_raw.get("header_footer_threshold", 0.3)
            ),
        ),
        enable_chunk_dedupe=bool(raw.get("enable_chunk_dedupe", False)),
        enable_semantic_dedupe=bool(raw.get("enable_semantic_dedupe", False)),
        semantic_dedupe_threshold=float(raw.get("semantic_dedupe_threshold", 0.85)),
        cleaners=raw.get("cleaners"),
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
            enable_multi_query=rewrite_raw.get("enable_multi_query", False),
            multi_query_count=int(rewrite_raw.get("multi_query_count", 3)),
        ),
    )
    cfg = _apply_rag_env_overrides(cfg)
    warn_model_instances(config_dir)
    return cfg
