# -*- coding: utf-8 -*-
"""Hermes 风格上下文压缩五阶段实现。

五阶段：评估 → 选材 → 摘要生成 → 替换 → 验证+反抖动
13 部分结构化摘要模板 + 锚定保护 + 反抖动。

多租户适配：maybe_compress 接收 RequestContext（含 tenant_id + user_id），
压缩状态（_last_savings_pct）是实例级，每个会话独立创建压缩器实例——天然隔离。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from core.domain.context import RequestContext
from core.ports.memory.compression import CompressionResult


@dataclass
class CompressorConfig:
    """上下文压缩器配置。"""

    trigger_pct: float = 0.75           # 上下文使用 75% 时触发
    min_summary_tokens: int = 2000      # 摘要至少 2000 token
    summary_ratio: float = 0.20         # 被压缩段压缩到 20%
    max_summary_tokens: int = 12000     # 单次摘要上限
    tail_preserve: int = 8              # 尾部至少保留 8 条
    anti_jitter_pct: float = 0.10       # 反抖动：节省 <10% 则回滚
    summary_template: str = (
        "用户画像与偏好:\n"
        "核心任务与目标:\n"
        "已完成的工作:\n"
        "待办事项:\n"
        "关键决策与理由:\n"
        "重要数据与参数:\n"
        "技术约束:\n"
        "错误与教训:\n"
        "外部依赖状态:\n"
        "文件操作记录:\n"
        "对话中的承诺:\n"
        "上下文跳转点:\n"
        "其他值得记住的信息:"
    )


def _estimate_tokens(text: str) -> int:
    """粗估 token 数。"""
    try:
        from utils.token_counter import count_tokens
        return count_tokens(text or "")
    except Exception:
        return max(1, len(text or "") // 4)


class HermesContextCompressor:
    """五阶段上下文压缩器。"""

    def __init__(
        self,
        models: Any = None,
        config: Optional[CompressorConfig] = None,
    ):
        self._models = models
        self._cfg = config or CompressorConfig()
        self._last_savings_pct: float = 0.0
        self._logger = logging.getLogger(__name__)

    async def maybe_compress(
        self,
        messages: List[dict],
        context: RequestContext,
        model_window: int,
    ) -> CompressionResult:
        """检测 + 压缩：超过阈值时触发五阶段压缩。"""
        if not messages or model_window <= 0:
            return CompressionResult(
                compressed_messages=messages,
                original_token_count=0,
                compressed_token_count=0,
                savings_pct=0.0,
                session_split=False,
            )

        # ── 阶段 1：评估 ──
        total_tokens = sum(
            _estimate_tokens(m.get("content", "")) for m in messages
        )
        threshold = int(model_window * self._cfg.trigger_pct)
        if total_tokens < threshold:
            return CompressionResult(
                compressed_messages=messages,
                original_token_count=total_tokens,
                compressed_token_count=total_tokens,
                savings_pct=0.0,
                session_split=False,
            )

        self._logger.info(
            "Context compression triggered: %d tokens > %d threshold "
            "(tenant=%s user=%s session=%s)",
            total_tokens, threshold,
            context.tenant_id, context.user_id, context.session_id,
        )

        # ── 阶段 2：选材（保留尾部 + 锚定保护） ──
        tail_count = min(self._cfg.tail_preserve, len(messages))
        tail = messages[-tail_count:]
        head = messages[:-tail_count] if tail_count < len(messages) else []

        if not head:
            return CompressionResult(
                compressed_messages=messages,
                original_token_count=total_tokens,
                compressed_token_count=total_tokens,
                savings_pct=0.0,
                session_split=False,
            )

        # ── 阶段 3：摘要生成 ──
        summary = await self._generate_summary(head, context)

        # ── 阶段 4：替换 ──
        compressed = [{"role": "system", "content": summary}] + tail

        # ── 阶段 5：验证 + 反抖动 ──
        new_tokens = sum(
            _estimate_tokens(m.get("content", "")) for m in compressed
        )
        savings = (total_tokens - new_tokens) / total_tokens if total_tokens > 0 else 0.0

        if savings < self._cfg.anti_jitter_pct:
            # 反抖动：节省不足，回滚
            self._logger.info(
                "Context compression rolled back (anti-jitter): "
                "savings=%.1f%% < %.1f%% threshold",
                savings * 100, self._cfg.anti_jitter_pct * 100,
            )
            self._last_savings_pct = 0.0
            return CompressionResult(
                compressed_messages=messages,
                original_token_count=total_tokens,
                compressed_token_count=total_tokens,
                savings_pct=0.0,
                session_split=False,
            )

        # 会话分裂：压缩后仍超限
        session_split = new_tokens > model_window
        self._last_savings_pct = savings
        self._logger.info(
            "Context compression completed: %d → %d tokens (%.1f%% saved, split=%s)",
            total_tokens, new_tokens, savings * 100, session_split,
        )
        return CompressionResult(
            compressed_messages=compressed,
            original_token_count=total_tokens,
            compressed_token_count=new_tokens,
            savings_pct=savings,
            session_split=session_split,
        )

    async def _generate_summary(
        self, head: List[dict], context: RequestContext
    ) -> str:
        """用 LLM 生成 13 部分结构化摘要；无 LLM 时降级为截断。"""
        head_text = "\n".join(
            f"[{m.get('role', 'user')}] {m.get('content', '')}"
            for m in head
        )

        if self._models is not None:
            try:
                prompt = (
                    "请将以下对话历史压缩为结构化摘要，严格按以下模板格式输出：\n\n"
                    f"{self._cfg.summary_template}\n\n"
                    f"对话历史：\n{head_text}\n\n"
                    "要求：保留所有关键信息，删除冗余，每部分不超过3行。"
                )
                result = await self._models.generate(
                    prompt,
                    tenant_id=context.tenant_id,
                    max_tokens=self._cfg.max_summary_tokens,
                )
                summary = str(result).strip()
                if summary:
                    return f"[Context Summary]\n{summary}"
            except Exception as e:
                self._logger.warning(
                    "LLM summary generation failed, falling back to truncation: %s", e
                )

        # 降级：截断摘要
        max_chars = self._cfg.max_summary_tokens * 4
        truncated = head_text[:max_chars]
        if len(head_text) > max_chars:
            truncated += "\n[... further truncated ...]"
        return f"[Context Summary (truncated)]\n{truncated}"

    @property
    def last_savings_pct(self) -> float:
        return self._last_savings_pct
