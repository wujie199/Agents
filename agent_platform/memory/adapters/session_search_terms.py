# -*- coding: utf-8 -*-
"""session_search 查询词切分（中文无空格 / 词序变化）。"""

from __future__ import annotations

import re
from typing import List


def extract_search_terms(query: str, *, max_terms: int = 10) -> List[str]:
    """从 query 提取 LIKE/FTS 子词；长中文串补充二字片段以提高命中率。"""
    q = (query or "").strip()
    if not q:
        return []

    seen: set[str] = set()
    terms: List[str] = []

    def add(term: str) -> None:
        t = term.strip()
        if len(t) < 2 or t in seen:
            return
        seen.add(t)
        terms.append(t)

    for part in re.split(r"[\s,，。！？、；;]+", q):
        if part.strip():
            add(part.strip())

    for m in re.finditer(r"[A-Za-z0-9_]{2,}", q):
        add(m.group())

    # 纯中文或混合长串：整句 + 二字 sliding window
    cjk = re.sub(r"[\s,，。！？、；;]+", "", q)
    if len(cjk) >= 2:
        add(cjk)
        if len(cjk) >= 4:
            for i in range(len(cjk) - 1):
                add(cjk[i : i + 2])
                if len(terms) >= max_terms:
                    break

    return terms[:max_terms] or [q]


def build_fts_match_query(query: str) -> str:
    """SQLite FTS MATCH 子句：空格分词 OR；长中文用二字 OR。"""
    terms = extract_search_terms(query, max_terms=12)
    if not terms:
        return '""'
    parts = []
    for t in terms:
        cleaned = t.replace('"', '""')
        parts.append(f'"{cleaned}"')
    return " OR ".join(parts)
