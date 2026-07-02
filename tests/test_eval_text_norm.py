# -*- coding: utf-8 -*-
"""Tests for shared eval text normalization."""

from document.rag.evaluation.text_norm import normalize_match_text


def test_normalize_strips_chinese_punctuation():
    a = normalize_match_text("驱动轮、万向轮，用镊子")
    b = normalize_match_text("驱动轮万向轮用镊子")
    assert a == b
