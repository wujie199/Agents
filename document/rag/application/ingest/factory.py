import logging
import mimetypes
from pathlib import Path
from typing import BinaryIO, Optional, Set

from core.ports.ingest import DocumentFormat, IngestConfig, IngestPort, IngestResult

from document.rag.adapters.ingest.layout_ocr_adapter import LayoutOCRAdapter
from document.rag.adapters.ingest.plain_text_adapter import PlainTextIngestAdapter
from document.rag.adapters.ingest.simplified_adapter import SimplifiedIngestAdapter
from document.rag.adapters.ingest.word_adapter import WordIngestAdapter
from document.rag.adapters.registry import build_ingest
from document.rag.config import RagPipelineConfig, load_rag_pipeline_config

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


def build_ingest_pipeline(config: Optional[RagPipelineConfig] = None) -> IngestPort:
    """按 config/rag_pipeline.yml ingest.mode 选择摄取适配器（见 adapters/registry）。"""
    return build_ingest(config)


def build_routed_ingest(cfg: RagPipelineConfig) -> IngestPort:
    plain = PlainTextIngestAdapter()
    word = WordIngestAdapter()
    ocr = LayoutOCRAdapter(
        ocr_backend=cfg.ingest.ocr_backend,
        language=cfg.ingest.language,
    )
    return RoutedIngestAdapter(
        plain_text_adapter=plain,
        word_adapter=word,
        layout_ocr_adapter=ocr,
        plain_text_formats={e.lower().lstrip(".") for e in cfg.ingest.plain_text_formats},
    )


class RoutedIngestAdapter:
    """按文件类型路由到结构化解析或 Layout OCR（structured 模式）。"""

    def __init__(
        self,
        plain_text_adapter: PlainTextIngestAdapter,
        word_adapter: WordIngestAdapter,
        layout_ocr_adapter: LayoutOCRAdapter,
        plain_text_formats: Optional[Set[str]] = None,
    ):
        self._plain = plain_text_adapter
        self._word = word_adapter
        self._ocr = layout_ocr_adapter
        self._plain_exts = plain_text_formats or {"txt", "md", "markdown"}
        self._logger = logging.getLogger("document.rag.ingest.routed")
        self._fallback = SimplifiedIngestAdapter(
            word_adapter=word_adapter,
            layout_ocr_adapter=layout_ocr_adapter,
        )

    def ingest(
        self,
        source: BinaryIO,
        doc_format: DocumentFormat,
        doc_id: str,
        config: Optional[IngestConfig] = None,
        metadata: Optional[dict] = None,
    ) -> IngestResult:
        if doc_format == DocumentFormat.WORD:
            self._logger.info("Ingest %s via Word adapter", doc_id)
            return self._word.ingest(source, doc_format, doc_id, config, metadata)
        if doc_format in (DocumentFormat.TEXT, DocumentFormat.MARKDOWN):
            self._logger.info("Ingest %s via PlainText adapter", doc_id)
            return self._plain.ingest(source, doc_format, doc_id, config, metadata)
        self._logger.info("Ingest %s via Layout OCR adapter", doc_id)
        return self._ocr.ingest(source, doc_format, doc_id, config, metadata)

    def ingest_from_path(
        self,
        file_path: str,
        doc_id: str,
        config: Optional[IngestConfig] = None,
        metadata: Optional[dict] = None,
    ) -> IngestResult:
        metadata = dict(metadata or {})
        metadata["source_path"] = file_path
        ext = Path(file_path).suffix.lower().lstrip(".")
        doc_format = detect_format(file_path)

        if ext in self._plain_exts:
            return self._plain.ingest_from_path(file_path, doc_id, config, metadata)
        if doc_format == DocumentFormat.WORD:
            with open(file_path, "rb") as f:
                return self._word.ingest(f, doc_format, doc_id, config, metadata)
        return self._fallback.ingest_from_path(file_path, doc_id, config, metadata)

    def supports_format(self, doc_format: DocumentFormat) -> bool:
        return doc_format in (
            DocumentFormat.WORD,
            DocumentFormat.PDF,
            DocumentFormat.IMAGE,
            DocumentFormat.HTML,
            DocumentFormat.TEXT,
            DocumentFormat.MARKDOWN,
        )
