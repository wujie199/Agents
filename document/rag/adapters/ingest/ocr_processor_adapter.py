"""通过 document/ocr/processor.py（UniversalOcrPipeline）进行统一 OCR 摄取。"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

from core.ports.ingest import (
    DocumentFormat,
    IngestConfig,
    IngestResult,
    IngestStatus,
)

from document.rag.adapters.ingest.word_to_pdf import (
    convert_to_pdf,
    needs_pdf_conversion,
)

logger = logging.getLogger("ingest.ocr_processor")

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
_PDF_EXTENSION = ".pdf"
_PLAIN_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}


class OcrProcessorIngestAdapter:
    """Word/PDF/图片/HTML 统一走 OCRProcessor；纯文本仍直接读取。"""

    def __init__(
        self,
        pdf_dpi: int = 200,
        use_layout: bool = True,
        word_to_pdf: bool = True,
        layout_model_dir: Optional[str] = None,
        ocr_model_dir: Optional[str] = None,
        model_root: Optional[str] = None,
        device: str = "cpu",
        preprocess_mode: str = "auto",
        enable_formula: bool = True,
        formula_model_name: Optional[str] = None,
        max_attempts: int = 3,
        fast_mode: bool = True,
        table_e2e: bool = False,
        enable_mkldnn: bool = True,
    ):
        self._pdf_dpi = pdf_dpi
        self._use_layout = use_layout
        self._word_to_pdf = word_to_pdf
        self._layout_model_dir = layout_model_dir
        self._ocr_model_dir = ocr_model_dir
        self._model_root = model_root or os.environ.get("OCR_MODEL_ROOT")
        self._device = device
        self._preprocess_mode = preprocess_mode
        self._enable_formula = enable_formula
        self._formula_model_name = formula_model_name
        self._max_attempts = max_attempts
        self._fast_mode = fast_mode
        self._table_e2e = table_e2e
        self._enable_mkldnn = enable_mkldnn
        self._processor = None
        self._temp_dirs: List[str] = []

    def supports_format(self, doc_format: DocumentFormat) -> bool:
        return doc_format in (
            DocumentFormat.WORD,
            DocumentFormat.PDF,
            DocumentFormat.IMAGE,
            DocumentFormat.HTML,
            DocumentFormat.TEXT,
            DocumentFormat.MARKDOWN,
        )

    def _get_processor(self):
        if self._processor is None:
            from document.ocr.processor import OCRProcessor

            kwargs: Dict[str, Any] = {
                "device": self._device,
                "preprocess_mode": self._preprocess_mode,
                "enable_formula": self._enable_formula,
                "max_attempts": self._max_attempts,
                "fast_mode": self._fast_mode,
                "table_e2e": self._table_e2e,
                "enable_mkldnn": self._enable_mkldnn,
            }
            if self._model_root:
                kwargs["model_root"] = self._model_root
            if self._layout_model_dir:
                kwargs["layout_model_dir"] = self._layout_model_dir
            if self._ocr_model_dir:
                kwargs["ocr_model_dir"] = self._ocr_model_dir
            if self._formula_model_name:
                kwargs["formula_model_name"] = self._formula_model_name
            self._processor = OCRProcessor(**kwargs)
        return self._processor

    def _cleanup_temp_dirs(self) -> None:
        for temp_dir in self._temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        self._temp_dirs.clear()

    def _detect_format(self, file_path: str) -> DocumentFormat:
        ext = Path(file_path).suffix.lower()
        if ext in (".doc", ".docx"):
            return DocumentFormat.WORD
        if ext == _PDF_EXTENSION:
            return DocumentFormat.PDF
        if ext in _IMAGE_EXTENSIONS:
            return DocumentFormat.IMAGE
        if ext in (".html", ".htm"):
            return DocumentFormat.HTML
        if ext in (".md", ".markdown"):
            return DocumentFormat.MARKDOWN
        return DocumentFormat.TEXT

    def _read_plain_text(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()

    def _prepare_ocr_path(
        self,
        file_path: str,
        doc_format: DocumentFormat,
    ) -> str:
        if doc_format in (DocumentFormat.TEXT, DocumentFormat.MARKDOWN):
            raise ValueError("plain text should not call _prepare_ocr_path")

        if self._word_to_pdf and needs_pdf_conversion(file_path):
            temp_dir = tempfile.mkdtemp(prefix="ingest_word_pdf_")
            self._temp_dirs.append(temp_dir)
            return convert_to_pdf(file_path, output_dir=temp_dir)

        if doc_format in (
            DocumentFormat.PDF,
            DocumentFormat.IMAGE,
            DocumentFormat.WORD,
            DocumentFormat.HTML,
        ):
            return file_path

        raise ValueError(f"Unsupported format for OCR: {doc_format}")

    def _ocr_result_to_ingest(
        self,
        ocr_result: Any,
        metadata: Dict[str, Any],
        source_path: str,
    ) -> IngestResult:
        pages: List[Dict[str, Any]] = []
        all_tables: List[Dict[str, Any]] = []

        if hasattr(ocr_result, "pages"):
            full_text = ocr_result.full_text
            for page in ocr_result.pages:
                page_tables = getattr(page, "tables", []) or []
                all_tables.extend(page_tables)
                page_meta = getattr(page, "metadata", {}) or {}
                pages.append(
                    {
                        "page_num": page.page_number,
                        "content": page.full_text,
                        "char_count": len(page.full_text),
                        "region_count": len(page.regions),
                        "table_count": len(page_tables),
                        "qc_status": page_meta.get("qc_status"),
                    }
                )
            metadata["total_pages"] = ocr_result.total_pages
            doc_meta = getattr(ocr_result, "metadata", {}) or {}
            if doc_meta.get("document_ir"):
                metadata["document_ir"] = doc_meta["document_ir"]
            if doc_meta.get("qc_summary"):
                metadata["qc_summary"] = doc_meta["qc_summary"]
        else:
            full_text = ocr_result.full_text
            all_tables = getattr(ocr_result, "tables", []) or []
            page_meta = getattr(ocr_result, "metadata", {}) or {}
            pages.append(
                {
                    "page_num": 1,
                    "content": full_text,
                    "char_count": len(full_text),
                    "region_count": len(ocr_result.regions),
                    "table_count": len(all_tables),
                    "qc_status": page_meta.get("qc_status"),
                }
            )
            metadata["total_pages"] = 1
            if page_meta.get("document_ir"):
                metadata["document_ir"] = page_meta["document_ir"]

        metadata["ingest_backend"] = "ocr_processor"
        metadata["ocr_pipeline"] = "UniversalOcrPipeline"
        metadata["ocr_source_path"] = source_path
        metadata["char_count"] = len(full_text)
        metadata["table_count"] = len(all_tables)
        if self._model_root:
            metadata["ocr_model_root"] = self._model_root

        status = IngestStatus.SUCCESS if full_text.strip() else IngestStatus.PARTIAL
        return IngestResult(
            content=full_text,
            metadata=metadata,
            status=status,
            pages=pages,
            tables=all_tables,
        )

    def ingest_from_path(
        self,
        file_path: str,
        doc_id: str,
        config: Optional[IngestConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestResult:
        config = config or IngestConfig()
        metadata = dict(metadata or {})
        metadata["doc_id"] = doc_id
        metadata["source_path"] = file_path

        doc_format = self._detect_format(file_path)
        metadata["doc_format"] = doc_format.value
        ext = Path(file_path).suffix.lower()

        try:
            if ext in _PLAIN_TEXT_EXTENSIONS or doc_format in (
                DocumentFormat.TEXT,
                DocumentFormat.MARKDOWN,
            ):
                content = self._read_plain_text(file_path)
                metadata["ingest_backend"] = "plain_text"
                metadata["ocr_skipped"] = True
                return IngestResult(
                    content=content,
                    metadata=metadata,
                    status=IngestStatus.SUCCESS if content else IngestStatus.PARTIAL,
                    pages=[
                        {
                            "page_num": 1,
                            "content": content,
                            "char_count": len(content),
                        }
                    ],
                )

            ocr_input = self._prepare_ocr_path(file_path, doc_format)
            dpi = config.dpi or self._pdf_dpi
            logger.info("OCR 摄取 %s via %s (dpi=%s)", doc_id, ocr_input, dpi)

            processor = self._get_processor()
            ocr_result = processor.process(
                ocr_input,
                use_layout=self._use_layout,
                use_ocr=True,
                pdf_dpi=dpi,
            )
            return self._ocr_result_to_ingest(ocr_result, metadata, file_path)

        except Exception as exc:
            logger.error("OCR 摄取失败 %s: %s", doc_id, exc)
            return IngestResult(
                content="",
                metadata=metadata,
                status=IngestStatus.FAILED,
                errors=[str(exc)],
            )
        finally:
            self._cleanup_temp_dirs()

    def ingest(
        self,
        source: BinaryIO,
        doc_format: DocumentFormat,
        doc_id: str,
        config: Optional[IngestConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestResult:
        metadata = dict(metadata or {})
        suffix_map = {
            DocumentFormat.WORD: ".docx",
            DocumentFormat.PDF: ".pdf",
            DocumentFormat.IMAGE: ".png",
            DocumentFormat.HTML: ".html",
            DocumentFormat.TEXT: ".txt",
            DocumentFormat.MARKDOWN: ".md",
        }
        suffix = suffix_map.get(doc_format, ".bin")
        temp_dir = tempfile.mkdtemp(prefix="ingest_upload_")
        temp_path = os.path.join(temp_dir, f"{doc_id}{suffix}")

        try:
            source.seek(0)
            with open(temp_path, "wb") as f:
                f.write(source.read())
            return self.ingest_from_path(temp_path, doc_id, config, metadata)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
