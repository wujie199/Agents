"""离线建库：摄取结果格式清理（与 KnowledgeBasePortAdapter 共用）。"""

import logging
from typing import Any, Optional

from core.ports.cleaner import CleaningLevel, DocumentType
from core.ports.ingest import DocumentFormat, IngestResult
from document.rag.components.cleaner.factory import build_cleaner_from_rag_config
from document.rag.components.cleaner.base import dedupe_header_footer_from_pages
from document.rag.config import RagPipelineConfig
from document.rag.shared.ocr_ingest_text import (
    apply_ocr_light_cleaning,
    rebuild_content_from_document_ir,
    sync_pages_content_from_ir,
)

logger = logging.getLogger("document.rag.application.cleaning.pipeline")

_FORMAT_TO_CLEANER_TYPE = {
    DocumentFormat.PDF: DocumentType.PDF,
    DocumentFormat.WORD: DocumentType.WORD,
    DocumentFormat.HTML: DocumentType.HTML,
    DocumentFormat.MARKDOWN: DocumentType.MARKDOWN,
    DocumentFormat.TEXT: DocumentType.TEXT,
    DocumentFormat.IMAGE: DocumentType.TEXT,
}

_CLEANING_LEVEL_MAP = {
    "light": CleaningLevel.LIGHT,
    "standard": CleaningLevel.STANDARD,
    "aggressive": CleaningLevel.AGGRESSIVE,
}


def parse_cleaning_level(level: str) -> CleaningLevel:
    return _CLEANING_LEVEL_MAP.get((level or "standard").lower(), CleaningLevel.STANDARD)


def _use_ocr_structure_preserving_clean(
    ingest_result: IngestResult,
    ingest_cfg: Any,
) -> bool:
    if ingest_result.metadata.get("ingest_backend") != "ocr_processor":
        return False
    if not getattr(ingest_cfg, "ocr_preserve_structure", True):
        return False
    return rebuild_content_from_document_ir(ingest_result.metadata) is not None


def apply_ingest_cleaning(
    ingest_result: IngestResult,
    doc_format: DocumentFormat,
    cfg: RagPipelineConfig,
    cleaner: Optional[Any] = None,
) -> int:
    """
    对摄取结果做 OCR 后处理 + CompositeCleaner。
    OCR + document_ir 时改用轻量清洗并统一从 IR 重建正文。
    返回清理后正文字符数；未启用清理时返回当前长度。
    """
    ingest_cfg = cfg.ingest
    if not ingest_cfg.enable_cleaning or not ingest_result.content:
        return len(ingest_result.content or "")

    doc_type = _FORMAT_TO_CLEANER_TYPE.get(doc_format, DocumentType.TEXT)
    backend = ingest_result.metadata.get("ingest_backend", "")
    raw_len = len(ingest_result.content)
    level = parse_cleaning_level(ingest_cfg.cleaning_level)
    cleaner_impl = cleaner or build_cleaner_from_rag_config(cfg)
    ocr_structure_mode = _use_ocr_structure_preserving_clean(ingest_result, ingest_cfg)

    if ingest_cfg.enable_header_footer_dedup and ingest_result.pages and not ocr_structure_mode:
        ingest_result.pages = dedupe_header_footer_from_pages(
            ingest_result.pages,
            threshold=ingest_cfg.header_footer_threshold,
        )
        ingest_result.content = "\n\n".join(
            (p.get("content") or "").strip()
            for p in ingest_result.pages
            if (p.get("content") or "").strip()
        )
        ingest_result.metadata["header_footer_deduped"] = True

    if ocr_structure_mode:
        ir_text = rebuild_content_from_document_ir(ingest_result.metadata)
        ingest_result.content = ir_text or ingest_result.content
        if ingest_result.pages:
            sync_pages_content_from_ir(ingest_result.pages, ingest_result.metadata)
        if ingest_cfg.ocr_postprocess:
            ingest_result.content = apply_ocr_light_cleaning(ingest_result.content)
            for page in ingest_result.pages or []:
                page_content = page.get("content", "")
                if page_content:
                    page["content"] = apply_ocr_light_cleaning(page_content)
        ingest_result.metadata["cleaned"] = True
        ingest_result.metadata["cleaning_mode"] = "ocr_structure_light"
        ingest_result.metadata["char_count"] = len(ingest_result.content)
        ingest_result.metadata["cleaning_level"] = "ocr_light"
        logger.info(
            "Cleaned %s (%s) OCR structure-preserving: %d -> %d chars",
            ingest_result.metadata.get("doc_id"),
            doc_format.value,
            raw_len,
            len(ingest_result.content),
        )
        return len(ingest_result.content)

    if ingest_cfg.ocr_postprocess and backend == "ocr_processor":
        from document.rag.shared.data_cleaner import postprocess_ocr

        ingest_result.content = postprocess_ocr(
            ingest_result.content,
            preserve_layout=True,
            preserve_tables=True,
        )
        for page in ingest_result.pages or []:
            page_content = page.get("content", "")
            if page_content:
                page["content"] = postprocess_ocr(
                    page_content,
                    preserve_layout=True,
                    preserve_tables=True,
                )

    ingest_result.content = cleaner_impl.clean(
        ingest_result.content,
        doc_type=doc_type,
        level=level,
        metadata=ingest_result.metadata,
    )
    for page in ingest_result.pages or []:
        page_content = page.get("content", "")
        if page_content:
            page["content"] = cleaner_impl.clean(
                page_content,
                doc_type=doc_type,
                level=level,
                metadata=ingest_result.metadata,
            )

    ingest_result.metadata["cleaned"] = True
    ingest_result.metadata["char_count"] = len(ingest_result.content)
    ingest_result.metadata["cleaning_level"] = level.value
    logger.info(
        "Cleaned %s (%s): %d -> %d chars",
        ingest_result.metadata.get("doc_id"),
        doc_format.value,
        raw_len,
        len(ingest_result.content),
    )
    return len(ingest_result.content)
