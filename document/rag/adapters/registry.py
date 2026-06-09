"""配置名 → 可替换适配器实现（ingest / embedding / rerank / metadata）。"""

from pathlib import Path
from typing import Any, Optional

from core.ports.ingest import IngestPort

from document.rag.config import RagPipelineConfig, load_rag_pipeline_config


def build_embedding(
    cfg: Optional[RagPipelineConfig] = None,
    *,
    config_dir: str = "config",
) -> Any:
    pipeline_cfg = cfg or load_rag_pipeline_config(config_dir=config_dir)
    backend = (pipeline_cfg.embedding.backend or "local_bge").lower()

    if backend == "mock":
        from document.rag.adapters.embedding.mock import MockEmbeddingModel

        return MockEmbeddingModel()

    if backend == "local_bge":
        from document.rag.adapters.embedding.local_bge import LocalBgeEmbedding

        emb = pipeline_cfg.embedding
        return LocalBgeEmbedding(
            model_dir=emb.model_path,
            device=emb.device,
            normalize_embeddings=emb.normalize,
        )

    raise ValueError(f"未知 embedding backend: {backend!r}")


def build_rerank(cfg: RagPipelineConfig) -> Any:
    if not cfg.retrieval.enable_rerank:
        return None

    rerank_raw = getattr(cfg, "rerank", None)
    backend = "local_bge"
    if rerank_raw is not None:
        backend = (rerank_raw.backend or "local_bge").lower()
    backend = backend.lower()

    if backend == "none":
        return None

    if backend == "mock":
        from document.rag.adapters.rerank.mock import MockRerankModel

        return MockRerankModel()

    if backend == "local_bge":
        from document.rag.adapters.rerank.local_bge import LocalBgeReranker

        r = rerank_raw
        return LocalBgeReranker(
            model_dir=getattr(r, "model_path", None) if r else None,
            device=getattr(r, "device", None) if r else None,
        )

    if cfg.retrieval.use_mock_rerank_fallback:
        from document.rag.adapters.rerank.mock import MockRerankModel

        return MockRerankModel()

    return None


def build_metadata_enricher(
    cfg: Optional[RagPipelineConfig] = None,
    *,
    config_dir: str = "config",
) -> Any:
    pipeline_cfg = cfg or load_rag_pipeline_config(config_dir=config_dir)
    meta = pipeline_cfg.metadata

    if not meta.enabled:
        from document.rag.adapters.metadata.none import NoOpMetadataEnricher

        return NoOpMetadataEnricher()

    backend = (meta.backend or "rule_keyword").lower()

    if backend == "none":
        from document.rag.adapters.metadata.none import NoOpMetadataEnricher

        return NoOpMetadataEnricher()

    if backend == "rule_keyword":
        from document.rag.adapters.metadata.rule_keyword import RuleKeywordMetadataEnricher

        rules_path = meta.rules_path or str(Path(config_dir) / "metadata_tagging.yml")
        return RuleKeywordMetadataEnricher(
            rules_path=rules_path,
            max_tags=meta.max_tags,
            tag_filename=meta.tag_filename,
        )

    raise ValueError(f"未知 metadata backend: {backend!r}")


def build_ingest(cfg: Optional[RagPipelineConfig] = None) -> IngestPort:
    pipeline_cfg = cfg or load_rag_pipeline_config()
    mode = (pipeline_cfg.ingest.mode or "ocr_only").lower()

    if mode == "ocr_only":
        from document.rag.adapters.ingest.ocr_processor_adapter import (
            OcrProcessorIngestAdapter,
        )

        ing = pipeline_cfg.ingest
        return OcrProcessorIngestAdapter(
            pdf_dpi=ing.pdf_dpi,
            use_layout=ing.ocr_use_layout,
            word_to_pdf=ing.word_to_pdf,
            model_root=ing.ocr_model_root,
            device=ing.ocr_device,
            preprocess_mode=ing.ocr_preprocess,
            enable_formula=ing.ocr_enable_formula,
            formula_model_name=ing.ocr_formula_model,
            max_attempts=ing.ocr_max_attempts,
            fast_mode=ing.ocr_fast,
            table_e2e=ing.ocr_table_e2e,
            enable_mkldnn=ing.ocr_enable_mkldnn,
        )

    from document.rag.application.ingest.factory import build_routed_ingest

    return build_routed_ingest(pipeline_cfg)


# 兼容旧名
resolve_local_embedding = build_embedding


def build_bm25_index(data_dir: Path, cfg: RagPipelineConfig) -> Any:
    from document.rag.adapters.retrieval.bm25_local import LocalBm25Index

    return LocalBm25Index.for_collection(data_dir, cfg.collection_name)
