"""摄取工厂 — detect_format + 按 config 构建 IngestPort（ocr_only）。"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from core.ports.ingest import DocumentFormat, IngestPort

from document.rag.components.ingest.registry import build_ingest
from document.rag.config import RagPipelineConfig

_EXT_TO_FORMAT = {
    "docx": DocumentFormat.WORD,
    "doc": DocumentFormat.WORD,
    "pdf": DocumentFormat.PDF,
    "png": DocumentFormat.IMAGE,
    "jpg": DocumentFormat.IMAGE,
    "jpeg": DocumentFormat.IMAGE,
    "bmp": DocumentFormat.IMAGE,
    "tiff": DocumentFormat.IMAGE,
    "tif": DocumentFormat.IMAGE,
    "webp": DocumentFormat.IMAGE,
    "html": DocumentFormat.HTML,
    "htm": DocumentFormat.HTML,
    "txt": DocumentFormat.TEXT,
    "md": DocumentFormat.MARKDOWN,
    "markdown": DocumentFormat.MARKDOWN,
}


def detect_format(path: str) -> DocumentFormat:
    ext = Path(path).suffix.lower().lstrip(".")
    if ext in _EXT_TO_FORMAT:
        return _EXT_TO_FORMAT[ext]
    mime, _ = mimetypes.guess_type(path)
    if mime and "word" in mime:
        return DocumentFormat.WORD
    if mime and "pdf" in mime:
        return DocumentFormat.PDF
    return DocumentFormat.TEXT


def build_ingest_pipeline(config: RagPipelineConfig) -> IngestPort:
    """按 config/rag.yml ingest.mode 选择摄取适配器（见 components/ingest/registry）。"""
    return build_ingest(config)
