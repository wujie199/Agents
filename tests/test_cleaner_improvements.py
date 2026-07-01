"""NFKC normalizer and header/footer dedup tests."""

import unicodedata

from document.rag.components.cleaner.base import (
    UnicodeNormalizerCleaner,
    dedupe_header_footer_from_pages,
)
from core.ports.cleaner import DocumentType, CleaningLevel


def test_unicode_normalizer_nfkc():
    cleaner = UnicodeNormalizerCleaner()
    # fullwidth digits -> ASCII
    raw = "ＡＢＣ１２３"
    out = cleaner.clean(raw, doc_type=DocumentType.TEXT, level=CleaningLevel.STANDARD)
    assert out == unicodedata.normalize("NFKC", raw)
    assert "ABC123" in out


def test_dedupe_header_footer_from_pages():
    pages = [
        {"page_num": 1, "content": "页眉标题\n正文A\n页脚-1"},
        {"page_num": 2, "content": "页眉标题\n正文B\n页脚-1"},
        {"page_num": 3, "content": "页眉标题\n正文C\n页脚-1"},
        {"page_num": 4, "content": "页眉标题\n正文D\n页脚-1"},
    ]
    cleaned = dedupe_header_footer_from_pages(pages, threshold=0.3)
    assert len(cleaned) == 4
    for page in cleaned:
        assert "页眉标题" not in page["content"]
        assert "页脚-1" not in page["content"]
        assert "正文" in page["content"]
