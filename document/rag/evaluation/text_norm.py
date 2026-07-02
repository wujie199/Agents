# -*- coding: utf-8 -*-
"""Shared text normalization for eval golden ↔ index chunk matching."""

from __future__ import annotations

import re

# 与建库清洗后文本对齐：去掉中英文标点、空白、常见 OCR/清洗 artifact
_MATCH_NORM_RE = re.compile(
    r"[\s\*：:，,。；;？?！!\-—、（）()【】\[\]「」""''《》·\.…]",
)


def normalize_match_text(text: str) -> str:
    return _MATCH_NORM_RE.sub("", (text or "").strip())


def strip_leading_faq_number(text: str) -> str:
    return re.sub(r"^\d{1,3}\.?", "", normalize_match_text(text))
