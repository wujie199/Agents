import pytest
from pathlib import Path

from core.ports.ingest import DocumentFormat
from document.rag.pipeline.ingest.factory import detect_format, build_ingest_pipeline


class TestIngestFactory:
    def test_detect_format(self):
        assert detect_format("a.docx") == DocumentFormat.WORD
        assert detect_format("b.pdf") == DocumentFormat.PDF
        assert detect_format("c.txt") == DocumentFormat.TEXT
        assert detect_format("d.md") == DocumentFormat.MARKDOWN

    def test_plain_text_ingest(self, tmp_path):
        p = tmp_path / "note.txt"
        p.write_text("plain text ingest test", encoding="utf-8")
        pipeline = build_ingest_pipeline()
        result = pipeline.ingest_from_path(str(p), doc_id="note1")
        assert result.status.value == "success"
        assert "plain text" in result.content
