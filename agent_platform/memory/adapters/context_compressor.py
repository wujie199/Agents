# -*- coding: utf-8 -*-
"""Hermes 风格上下文压缩五阶段实现。

五阶段：评估 → prune → 选材 → 摘要生成 → 替换 → 验证+反抖动
13 部分结构化摘要模板 + 锚定保护 + 反抖动 + tool pair 修复。

L0 压缩仅改写对话 messages（summary+tail），不会 bulk 写入 L3 skills 到 L1；
技能生命周期由 L3 finalize / on_session_end 草稿抽取负责。

多租户适配：maybe_compress 接收 RequestContext（含 tenant_id + user_id），
压缩状态（_last_savings_pct）是实例级，每个会话独立创建压缩器实例——天然隔离。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from core.domain.context import RequestContext
from core.ports.memory.compression import CompressionResult

from agent_platform.memory.adapters.context_window_manager import (
    ContextWindowManager,
    prune_old_tool_results,
    repair_tool_message_pairs,
)


@dataclass
class CompressorConfig:
    """上下文压缩器配置。"""

    enabled: bool = True
    trigger_pct: float = 0.50           # 上下文使用 50% 时触发
    compress_target_ratio: float = 0.20   # tail 预算 = threshold × ratio
    min_summary_tokens: int = 2000
    summary_ratio: float = 0.20
    max_summary_tokens: int = 12000
    tail_preserve: int = 8              # 最少保留消息条数（下限）
    tool_prune_min_chars: int = 200
    tool_prune_placeholder: str = "[tool result truncated]"
    anti_jitter_pct: float = 0.10
    context_window_tokens: int = 128000
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


def _estimate_messages_tokens(messages: List[dict]) -> int:
    mgr = ContextWindowManager(_estimate_tokens)
    total = 0
    for msg in messages:
        total += _message_tokens(msg, mgr._estimate)
    return total


def _message_tokens(msg: dict, estimate_fn) -> int:
    content = msg.get("content", "") or ""
    tokens = estimate_fn(str(content))
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            tokens += estimate_fn(str(fn.get("name", "")))
            tokens += estimate_fn(str(fn.get("arguments", "")))
    return tokens


def compressor_config_from_dict(cfg: dict[str, Any]) -> CompressorConfig:
    """从 memory.yml / chat.yml 扁平键构建 CompressorConfig。"""
    return CompressorConfig(
        enabled=bool(cfg.get("l0_context_compress_enabled", True)),
        trigger_pct=float(
            cfg.get("context_compress_threshold", cfg.get("l0_context_compress_threshold", 0.50))
        ),
        compress_target_ratio=float(
            cfg.get("compress_target_ratio", cfg.get("l0_compress_target_ratio", 0.20))
        ),
        min_summary_tokens=int(cfg.get("l0_min_summary_tokens", 2000)),
        summary_ratio=float(cfg.get("l0_summary_ratio", 0.20)),
        max_summary_tokens=int(cfg.get("l0_max_summary_tokens", 12000)),
        tail_preserve=int(cfg.get("l0_tail_preserve_min", 8)),
        tool_prune_min_chars=int(cfg.get("l0_tool_prune_min_chars", 200)),
        tool_prune_placeholder=str(
            cfg.get("l0_tool_prune_placeholder", "[tool result truncated]")
        ),
        anti_jitter_pct=float(cfg.get("l0_anti_jitter_pct", 0.10)),
        context_window_tokens=int(
            cfg.get("context_window_tokens", cfg.get("l0_context_window_tokens", 128000))
        ),
    )


def build_context_compressor(
    mem_cfg: dict[str, Any],
    models: Any = None,
) -> HermesContextCompressor:
    """工厂：从 memory 配置构建 L0 压缩器。"""
    return HermesContextCompressor(
        models=models,
        config=compressor_config_from_dict(mem_cfg),
    )


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
        self._window_mgr = ContextWindowManager(_estimate_tokens)

    @property
    def config(self) -> CompressorConfig:
        return self._cfg

    def resolve_model_window(self, model_window: Optional[int] = None) -> int:
        window = model_window if model_window and model_window > 0 else None
        return window or self._cfg.context_window_tokens

    async def maybe_compress(
        self,
        messages: List[dict],
        context: RequestContext,
        model_window: int,
        *,
        prompt_tokens: Optional[int] = None,
        provider_context: str = "",
    ) -> CompressionResult:
        """检测 + 压缩：超过阈值时触发五阶段压缩。"""
        if not self._cfg.enabled:
            return self._no_op(messages)

        resolved_window = self.resolve_model_window(model_window)
        if not messages or resolved_window <= 0:
            return self._no_op(messages, model_window=resolved_window)

        # ── 阶段 1：评估 ──
        estimated = _estimate_messages_tokens(messages)
        total_tokens = int(prompt_tokens) if prompt_tokens and prompt_tokens > 0 else estimated
        threshold = int(resolved_window * self._cfg.trigger_pct)
        if total_tokens < threshold:
            return CompressionResult(
                compressed_messages=messages,
                original_token_count=total_tokens,
                compressed_token_count=total_tokens,
                savings_pct=0.0,
                session_split=False,
                triggered=False,
                pruned_tool_results=0,
            )

        self._logger.info(
            "Context compression triggered: %d tokens > %d threshold "
            "(tenant=%s user=%s session=%s)",
            total_tokens,
            threshold,
            context.tenant_id,
            context.user_id,
            context.session_id,
        )

        tail_budget = max(1, int(threshold * self._cfg.compress_target_ratio))
        pruned_full, pruned_count = prune_old_tool_results(
            messages,
            tail_start=0,
            min_chars=self._cfg.tool_prune_min_chars,
            placeholder=self._cfg.tool_prune_placeholder,
        )
        # 先分区再修正 tail 外 prune（按 tail 边界重算）
        zones_after = self._window_mgr.split_zones(
            pruned_full, tail_token_budget=tail_budget
        )
        tail_start = self._window_mgr.tail_start_index(pruned_full, zones_after)
        pruned_full, pruned_count = prune_old_tool_results(
            pruned_full,
            tail_start=tail_start,
            min_chars=self._cfg.tool_prune_min_chars,
            placeholder=self._cfg.tool_prune_placeholder,
        )
        zones_after = self._window_mgr.split_zones(
            pruned_full, tail_token_budget=tail_budget
        )

        middle = zones_after.middle
        if not middle:
            return CompressionResult(
                compressed_messages=messages,
                original_token_count=total_tokens,
                compressed_token_count=total_tokens,
                savings_pct=0.0,
                session_split=False,
                triggered=True,
                pruned_tool_results=pruned_count,
            )

        # ── 阶段 3：摘要生成 ──
        summary = await self._generate_summary(
            middle, context, provider_context=provider_context
        )

        # ── 阶段 4：替换 ──
        summary_msg = {"role": "system", "content": summary}
        compressed = self._window_mgr.reassemble(
            zones_after.protected_head,
            summary_msg,
            zones_after.tail,
        )
        compressed = repair_tool_message_pairs(compressed)

        # ── 阶段 5：验证 + 反抖动 ──
        new_tokens = _estimate_messages_tokens(compressed)
        savings = (total_tokens - new_tokens) / total_tokens if total_tokens > 0 else 0.0

        if savings < self._cfg.anti_jitter_pct:
            self._logger.info(
                "Context compression rolled back (anti-jitter): "
                "savings=%.1f%% < %.1f%% threshold",
                savings * 100,
                self._cfg.anti_jitter_pct * 100,
            )
            self._last_savings_pct = 0.0
            return CompressionResult(
                compressed_messages=messages,
                original_token_count=total_tokens,
                compressed_token_count=total_tokens,
                savings_pct=0.0,
                session_split=False,
                triggered=True,
                pruned_tool_results=pruned_count,
            )

        session_split = new_tokens > resolved_window
        self._last_savings_pct = savings
        self._logger.info(
            "Context compression completed: %d → %d tokens (%.1f%% saved, split=%s)",
            total_tokens,
            new_tokens,
            savings * 100,
            session_split,
        )
        return CompressionResult(
            compressed_messages=compressed,
            original_token_count=total_tokens,
            compressed_token_count=new_tokens,
            savings_pct=savings,
            session_split=session_split,
            triggered=True,
            pruned_tool_results=pruned_count,
        )

    def _no_op(
        self,
        messages: List[dict],
        *,
        model_window: Optional[int] = None,
    ) -> CompressionResult:
        tokens = _estimate_messages_tokens(messages) if messages else 0
        return CompressionResult(
            compressed_messages=messages,
            original_token_count=tokens,
            compressed_token_count=tokens,
            savings_pct=0.0,
            session_split=False,
            triggered=False,
            pruned_tool_results=0,
        )

    async def _generate_summary(
        self,
        head: List[dict],
        context: RequestContext,
        *,
        provider_context: str = "",
    ) -> str:
        """用 LLM 生成 13 部分结构化摘要；无 LLM 时降级为截断。"""
        head_text = "\n".join(
            f"[{m.get('role', 'user')}] {m.get('content', '')}"
            for m in head
        )
        provider_block = ""
        if provider_context and provider_context.strip():
            provider_block = (
                f"\n\n外部记忆补充（压缩时合并）：\n{provider_context.strip()}\n"
            )

        if self._models is not None:
            try:
                prompt = (
                    "请将以下对话历史压缩为结构化摘要，严格按以下模板格式输出：\n\n"
                    f"{self._cfg.summary_template}\n\n"
                    f"对话历史：\n{head_text}\n\n"
                    f"{provider_block}"
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

        max_chars = self._cfg.max_summary_tokens * 4
        truncated = head_text[:max_chars]
        if len(head_text) > max_chars:
            truncated += "\n[... further truncated ...]"
        return f"[Context Summary (truncated)]\n{truncated}"

    @property
    def last_savings_pct(self) -> float:
        return self._last_savings_pct
