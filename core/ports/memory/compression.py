# -*- coding: utf-8 -*-
"""上下文压缩端口 — Hermes 五阶段压缩契约。"""

from __future__ import annotations

from typing import Protocol, List

from dataclasses import dataclass


@dataclass(frozen=True)
class CompressionResult:
    """上下文压缩结果。"""

    compressed_messages: List[dict]
    original_token_count: int
    compressed_token_count: int
    savings_pct: float
    session_split: bool


class ContextCompressorPort(Protocol):
    """上下文压缩契约（Hermes 横切层）。

    多租户适配：maybe_compress 接收 RequestContext（含 tenant_id + user_id），
    摘要生成的 LLM 调用带租户上下文。
    """

    async def maybe_compress(
        self,
        messages: List[dict],
        context: "core.domain.context.RequestContext",
        model_window: int,
    ) -> CompressionResult:
        """检测 + 压缩：超过阈值时触发五阶段压缩。"""
        ...
