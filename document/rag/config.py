from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class IngestConfig:
    routing: str = "simplified"
    mode: str = "ocr_only"
    plain_text_formats: List[str] = field(default_factory=lambda: ["txt", "md"])
    ocr_backend: str = "auto"
    language: str = "ch"
    word_to_pdf: bool = True
    word_converter: str = "libreoffice"
    pdf_dpi: int = 200
    ocr_use_layout: bool = True
    enable_cleaning: bool = True
    ocr_postprocess: bool = True
    cleaning_level: str = "standard"


@dataclass
class RetrievalConfig:
    primary_backend: str = "vector"
    enable_rerank: bool = False
    enable_router: bool = True
    auto_route: bool = True
    enable_graph: bool = False
    enable_sql: bool = False
    use_mock_rerank_fallback: bool = True
    enable_hybrid: bool = False
    enable_vector_search: bool = True
    enable_bm25_search: bool = True
    vector_top_k: int = 10
    bm25_top_k: int = 10
    hybrid_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])
    fusion_strategy: str = "weighted"
    fusion_top_n: int = 10
    rerank_top_n: int = 5
    rerank_min_score: Optional[float] = 0.8


@dataclass
class RewriteConfig:
    enable_hyde: bool = False
    enable_multi_query: bool = False
    multi_query_count: int = 3


@dataclass
class EmbeddingConfig:
    """Embedding 适配器（backend 见 adapters/registry）。"""

    backend: str = "local_bge"
    model_path: Optional[str] = None
    device: Optional[str] = None
    normalize: bool = True


@dataclass
class RerankConfig:
    """Rerank 适配器（backend: local_bge | mock | none）。"""

    backend: str = "local_bge"
    model_path: Optional[str] = None
    device: Optional[str] = None


@dataclass
class MetadataConfig:
    """元数据打标（backend 见 adapters/registry.build_metadata_enricher）。"""

    enabled: bool = True
    backend: str = "rule_keyword"
    rules_path: Optional[str] = None
    max_tags: int = 32
    tag_filename: bool = True


@dataclass
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


def load_rag_pipeline_config(
    config_path: Optional[str] = None,
    config_dir: str = "config",
) -> RagPipelineConfig:
    if config_path is None:
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
    chroma: dict = {}
    if chroma_path.exists():
        with open(chroma_path, "r", encoding="utf-8") as f:
            chroma = yaml.safe_load(f) or {}
    if path.name == "rag_pipeline.yml" and not raw.get("collection_name"):
        collection_name = chroma.get("collection_name", collection_name)

    return RagPipelineConfig(
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
        ),
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
