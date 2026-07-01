"""OCR 摄取适配器与 ocr_only 模式测试。"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from document.rag.config import IngestConfig as PipelineIngestConfig, RagPipelineConfig
from document.rag.components.ingest.ocr_processor import OcrProcessorIngestAdapter
from document.rag.components.ingest.word_to_pdf import needs_pdf_conversion
from document.rag.application.ingest_factory import build_ingest_pipeline, detect_format
from core.ports.ingest import DocumentFormat, IngestConfig, IngestStatus


class TestWordToPdfHelpers:
    def test_needs_pdf_conversion(self):
        assert needs_pdf_conversion("a.docx") is True
        assert needs_pdf_conversion("b.html") is True
        assert needs_pdf_conversion("c.pdf") is False


class TestOcrProcessorIngestAdapter:
    def test_plain_text_skips_ocr(self, tmp_path):
        p = tmp_path / "note.txt"
        p.write_text("纯文本直接读取", encoding="utf-8")
        adapter = OcrProcessorIngestAdapter()
        result = adapter.ingest_from_path(str(p), doc_id="t1")
        assert result.status == IngestStatus.SUCCESS
        assert result.content == "纯文本直接读取"
        assert result.metadata.get("ocr_skipped") is True

    @patch("document.rag.components.ingest.ocr_processor.OcrProcessorIngestAdapter._get_processor")
    def test_pdf_uses_ocr_processor(self, mock_get_processor, tmp_path):
        mock_processor = MagicMock()
        mock_result = MagicMock()
        mock_result.full_text = "OCR 识别内容"
        mock_result.regions = []
        mock_get_processor.return_value = mock_processor
        mock_processor.process.return_value = mock_result

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 mock")

        adapter = OcrProcessorIngestAdapter()
        result = adapter.ingest_from_path(str(pdf), doc_id="pdf1")

        assert result.status == IngestStatus.SUCCESS
        assert result.content == "OCR 识别内容"
        assert result.metadata.get("ingest_backend") == "ocr_processor"
        mock_processor.process.assert_called_once()

    @patch("document.rag.components.ingest.ocr_processor.convert_to_pdf")
    @patch("document.rag.components.ingest.ocr_processor.OcrProcessorIngestAdapter._get_processor")
    def test_word_converts_to_pdf_then_ocr(
        self, mock_get_processor, mock_convert, tmp_path
    ):
        docx = tmp_path / "contract.docx"
        docx.write_bytes(b"PK mock docx")
        pdf_path = tmp_path / "contract.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")
        mock_convert.return_value = str(pdf_path)

        mock_processor = MagicMock()
        mock_pdf_result = MagicMock()
        mock_pdf_result.full_text = "租房协议"
        mock_pdf_result.total_pages = 1
        mock_page = MagicMock()
        mock_page.page_number = 1
        mock_page.full_text = "租房协议"
        mock_page.regions = []
        mock_pdf_result.pages = [mock_page]
        mock_get_processor.return_value = mock_processor
        mock_processor.process.return_value = mock_pdf_result

        adapter = OcrProcessorIngestAdapter(word_to_pdf=True)
        result = adapter.ingest_from_path(str(docx), doc_id="word1")

        mock_convert.assert_called_once()
        assert result.status == IngestStatus.SUCCESS
        assert "租房协议" in result.content


class TestIngestFactoryOcrOnly:
    def test_build_ocr_only_pipeline(self):
        cfg = RagPipelineConfig(
            ingest=PipelineIngestConfig(mode="ocr_only"),
        )
        pipeline = build_ingest_pipeline(cfg)
        assert isinstance(pipeline, OcrProcessorIngestAdapter)

    def test_build_structured_pipeline(self):
        cfg = RagPipelineConfig(
            ingest=PipelineIngestConfig(mode="structured"),
        )
        pipeline = build_ingest_pipeline(cfg)
        from document.rag.application.ingest_factory import RoutedIngestAdapter

        assert isinstance(pipeline, RoutedIngestAdapter)

    def test_plain_text_in_ocr_only_mode(self, tmp_path):
        cfg = RagPipelineConfig(ingest=PipelineIngestConfig(mode="ocr_only"))
        p = tmp_path / "note.txt"
        p.write_text("ocr_only plain", encoding="utf-8")
        pipeline = build_ingest_pipeline(cfg)
        result = pipeline.ingest_from_path(str(p), doc_id="n1")
        assert result.status.value == "success"
        assert "ocr_only plain" in result.content

    def test_detect_format_word(self):
        assert detect_format("x.docx") == DocumentFormat.WORD
