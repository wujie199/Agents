"""Ingest 组件 registry — 当前仅支持 ocr_only。"""

from __future__ import annotations

import logging

from core.ports.ingest import IngestPort
from document.rag.config.pipeline import RagPipelineConfig

_log = logging.getLogger("document.rag.components.ingest.registry")


def build_ingest(cfg: RagPipelineConfig) -> IngestPort:
    mode = (cfg.ingest.mode or "ocr_only").lower()
    if mode not in ("ocr_only", "ocr"):
        _log.warning("ingest.mode=%s 已废弃，回退 ocr_only", mode)

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
        layout_threshold=ing.ocr_layout_threshold,
        layout_score_threshold=ing.ocr_layout_score_threshold,
        fast_mode=ing.ocr_fast,
        table_e2e=ing.ocr_table_e2e,
        enable_mkldnn=ing.ocr_enable_mkldnn,
        enable_pdf_routing=ing.enable_pdf_routing,
        pdf_threads=ing.pdf_threads,
    )
