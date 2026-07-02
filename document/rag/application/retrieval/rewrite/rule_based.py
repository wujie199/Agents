# -*- coding: utf-8 -*-
"""规则 Query 改写：L0 归一化 + 按 profile 扩展（无需 LLM）。"""

from __future__ import annotations

import logging
from typing import List, Optional

from document.rag.application.retrieval.query_intent import (
    detect_maintenance_intent,
    maintenance_entity_suffix,
)
from document.rag.application.retrieval.rewrite.rewrite_profile import (
    normalize_query,
    resolve_rewrite_profile,
)

_logger = logging.getLogger("rag.rewrite.rule_based")

_COMPARE_TERMS = ("对比", "区别", "参数", "优缺点")


class RuleBasedQueryRewriter:
    """检索前规则改写，对接 QueryRewritePort。"""

    def __init__(self, *, max_queries: int = 4) -> None:
        self._max_queries = max(1, max_queries)

    async def rewrite(
        self,
        query: str,
        *,
        profile: Optional[str] = None,
    ) -> List[str]:
        q = (query or "").strip()
        if not q:
            return [q]

        prof = profile or resolve_rewrite_profile(q)
        stripped = normalize_query(q)

        if prof == "exact_lookup":
            return [q]

        if prof == "generic_knowledge":
            return self._dedupe([q, stripped] if stripped != q else [q])

        if prof == "faq_like":
            return self._dedupe([q, stripped] if stripped and stripped != q else [q])

        if prof == "how_to":
            variants = [q]
            if stripped and stripped != q:
                variants.append(stripped)
            if "步骤" not in stripped:
                variants.append(f"{stripped or q} 操作步骤")
            return self._dedupe(variants)

        if prof == "product_compare":
            variants = [q]
            if stripped and stripped != q:
                variants.append(stripped)
            core = stripped or q
            if not any(t in core for t in _COMPARE_TERMS):
                variants.append(f"{core} 对比 参数")
            return self._dedupe(variants)

        if prof == "maintenance":
            return self._dedupe(self._maintenance_variants(q, stripped))

        return self._dedupe([q, stripped] if stripped != q else [q])

    def _maintenance_variants(self, q: str, stripped: str) -> List[str]:
        variants: List[str] = []
        if stripped and stripped != q:
            variants.append(stripped)
        suffix = maintenance_entity_suffix(q)
        core = stripped or q
        if suffix:
            variants.append(f"{core} {suffix}".strip())
        variants.extend(
            [
                "清洁机器人机身 断开电源 不可用水冲洗 干布擦拭",
                "扫地机器人 维护保养 尘盒 主刷 边刷 滤网 传感器",
                "每日使用后 擦拭机器人机身外壳 去除灰尘水渍",
            ]
        )
        return [q, *variants]

    def _dedupe(self, candidates: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for candidate in candidates:
            c = candidate.strip()
            if not c or c in seen:
                continue
            seen.add(c)
            out.append(c)
            if len(out) >= self._max_queries:
                break
        if len(out) > 1:
            _logger.debug("rule rewrite -> %d queries", len(out))
        return out

    async def rewrite_batch(self, queries: List[str]) -> List[List[str]]:
        import asyncio

        return await asyncio.gather(*[self.rewrite(q) for q in queries])

    def is_enabled(self) -> bool:
        return True
