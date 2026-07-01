"""Ingest 组件独立 registry — 按 mode 配置构建 IngestPort。"""

from core.ports.ingest import IngestPort
from document.rag.config.pipeline import RagPipelineConfig


def build_ingest(cfg: RagPipelineConfig) -> IngestPort:
    mode = (cfg.ingest.mode or "ocr_only").lower()

    if mode == "ocr_only":
        from document.rag.components.ingest.ocr_processor import OcrProcessorIngestAdapter
        ing = cfg.ingest
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

    from document.rag.application.ingest_factory import build_routed_ingest
    return build_routed_ingest(cfg)
