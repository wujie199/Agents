"""兼容 re-export（含 patch 用的模块级 convert_to_pdf）。"""
from document.rag.adapters.ingest.ocr_processor_adapter import (  # noqa: F401
    OcrProcessorIngestAdapter,
    convert_to_pdf,
)

__all__ = ["OcrProcessorIngestAdapter", "convert_to_pdf"]
