# -*- coding: utf-8 -*-
"""Query 改写画像：L0 归一化 + profile 识别。"""

from __future__ import annotations

import re

from document.rag.application.retrieval.query_intent import (
    detect_maintenance_intent,
    has_faq_template_in_query,
    strip_query_templates,
)

RewriteProfile = str

_EXACT_LOOKUP_RE = re.compile(
    r"^[A-Z0-9]{2,}[-_]?\d{2,}$|^\d{3,}$",
    re.I,
)
_HOW_TO_RE = re.compile(r"如何|怎么(?:样|做|办)?|怎样|步骤|教程")
_COMPARE_RE = re.compile(r"对比|比较|区别|差异|哪个好|哪款|推荐哪个|选哪个")


def normalize_query(query: str) -> str:
    """L0：模板剥离 + 空白归一（所有 RAG query 可用）。"""
    q = (query or "").strip()
    if not q:
        return q
    stripped = strip_query_templates(q)
    return stripped or q


def resolve_rewrite_profile(
    query: str,
    *,
    default: str = "generic_knowledge",
) -> RewriteProfile:
    """按 query 选择改写 profile（优先级：精确查找 > 维护 > 对比 > 操作 > FAQ 模板 > 默认）。"""
    q = (query or "").strip()
    if not q:
        return default
    if _EXACT_LOOKUP_RE.match(q.replace(" ", "")):
        return "exact_lookup"
    if detect_maintenance_intent(q):
        return "maintenance"
    if _COMPARE_RE.search(q):
        return "product_compare"
    if _HOW_TO_RE.search(q):
        return "how_to"
    if has_faq_template_in_query(q):
        return "faq_like"
    return default


def cap_queries(queries: list[str], max_n: int) -> list[str]:
    if max_n <= 0:
        return queries
    if len(queries) <= max_n:
        return queries
    return queries[:max_n]
