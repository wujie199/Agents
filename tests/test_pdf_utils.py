"""pypdfium2 PDF 渲染资源管理测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from document.ocr.pdf_native import extract_native_pages
from document.ocr.pdf_utils import pdf_to_images

PDF = Path(__file__).resolve().parents[2] / "data" / "test_docs" / "扫地机器人100问.pdf"


@pytest.mark.skipif(not PDF.is_file(), reason="test PDF missing")
def test_pdf_to_images_closes_document():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        paths = pdf_to_images(PDF, out, dpi=150, pdf_threads=1)
        assert len(paths) == 8
        assert all(p.is_file() for p in paths)


@pytest.mark.skipif(not PDF.is_file(), reason="test PDF missing")
def test_extract_native_pages_batch():
    pages = extract_native_pages(PDF, [0, 1, 2])
    assert set(pages) == {0, 1, 2}
    assert all(pages[i]["full_text"] for i in (0, 1, 2))
