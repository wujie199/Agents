# -*- coding: utf-8 -*-
"""组合 Query 改写：profile 策略 + L1 规则 + 可选 L2 LLM。"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from document.rag.application.retrieval.rewrite.multi_query import QueryRewriterPipeline
from document.rag.application.retrieval.rewrite.rewrite_profile import (
    cap_queries,
    normalize_query,
    resolve_rewrite_profile,
)
from document.rag.config.rewrite import RewriteConfig, RewriteProfilePolicy

_logger = logging.getLogger("rag.rewrite.combined")

LlmRewriteMode = str  # never | on_miss | always


class CombinedQueryRewriter:
    """L0 归一化 + profile 化 L1 规则 + 按策略 L2 LLM。"""

    def __init__(
        self,
        rule_rewriter: Any,
        llm_pipeline: Optional[QueryRewriterPipeline] = None,
        *,
        llm_rewrite_once: bool = True,
        rewrite_config: Optional[RewriteConfig] = None,
    ) -> None:
        self._rule = rule_rewriter
        self._llm = llm_pipeline
        self._llm_rewrite_once = llm_rewrite_once
        self._cfg = rewrite_config

    def _policy(self, profile: str) -> RewriteProfilePolicy:
        if self._cfg is not None:
            return self._cfg.profile_policy(profile)
        from document.rag.config.rewrite import DEFAULT_REWRITE_PROFILES

        return DEFAULT_REWRITE_PROFILES.get(
            profile,
            DEFAULT_REWRITE_PROFILES["generic_knowledge"],
        )

    async def rewrite(
        self,
        query: str,
        *,
        llm_mode: Optional[LlmRewriteMode] = None,
        profile: Optional[str] = None,
    ) -> List[str]:
        original = (query or "").strip()
        if not original:
            return [query]

        prof = profile or resolve_rewrite_profile(
            original,
            default=(self._cfg.default_profile if self._cfg else "generic_knowledge"),
        )
        policy = self._policy(prof)
        effective_llm = llm_mode if llm_mode is not None else policy.llm

        merged: List[str] = [original]

        if policy.rule and self._rule is not None and getattr(
            self._rule, "is_enabled", lambda: False
        )():
            try:
                merged = await self._rule.rewrite(original, profile=prof)
            except (RuntimeError, ValueError, TypeError) as exc:
                _logger.warning("Rule rewrite failed: %s", exc)
                merged = [original]
        else:
            norm = normalize_query(original)
            merged = [original] if norm == original else [original, norm]

        if effective_llm == "always" and self._llm is not None and self._llm.is_enabled():
            merged = await self._append_llm_queries(original, merged)

        unique = list(dict.fromkeys(q.strip() for q in merged if q and q.strip()))
        capped = cap_queries(unique or [original], policy.max_hybrid_queries)
        return capped or [original]

    async def expand_llm(self, query: str) -> List[str]:
        """仅 LLM 扩写（两段式第 2 轮）。"""
        original = (query or "").strip()
        if not original or self._llm is None or not self._llm.is_enabled():
            return []
        try:
            expanded = await self._llm.rewrite(original)
            return [q for q in expanded if q and q.strip() and q.strip() != original]
        except Exception as exc:
            _logger.warning("LLM expand failed: %s", exc)
            return []

    async def _append_llm_queries(
        self, original: str, merged: List[str]
    ) -> List[str]:
        if self._llm is None or not self._llm.is_enabled():
            return merged
        targets = [original] if self._llm_rewrite_once else list(merged)
        out = list(merged)
        for target in targets:
            try:
                out.extend(await self._llm.rewrite(target))
            except Exception as exc:
                _logger.warning("LLM rewrite failed for %r: %s", target[:40], exc)
                out.append(target)
        return out

    async def rewrite_batch(self, queries: List[str]) -> List[List[str]]:
        import asyncio

        return await asyncio.gather(*[self.rewrite(q) for q in queries])

    def is_enabled(self) -> bool:
        rule_on = self._rule is not None and getattr(
            self._rule, "is_enabled", lambda: False
        )()
        llm_on = self._llm is not None and self._llm.is_enabled()
        return rule_on or llm_on
