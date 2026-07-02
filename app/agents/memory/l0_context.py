# -*- coding: utf-8 -*-
"""L0 上下文压缩：对话循环集成（prepare 注入 + turn 后评估）。"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, List, Optional

from core.composition.run_context import RunContext
from core.ports.observability import Layer

from agent_platform.memory.adapters.context_compressor import (
    CompressorConfig,
    HermesContextCompressor,
    compressor_config_from_dict,
)
from app.agents.orchestration.chat_config import ChatAgentConfig
from app.agents.observability.graph_metrics import record_l0_compress_triggered
from app.agents.observability.instrument import span_ctx

_logger = logging.getLogger(__name__)

_L0_STATE_KEY = "_l0_context_state"
_SUMMARY_MARKER = "[Context Summary"


def _extra(ctx: RunContext) -> dict:
    extra = getattr(ctx, "extra", None)
    if not isinstance(extra, dict):
        return {}
    return extra


def get_context_compressor(ctx: RunContext) -> Optional[HermesContextCompressor]:
    comp = _extra(ctx).get("context_compressor")
    return comp if isinstance(comp, HermesContextCompressor) else None


def resolve_l0_config(ctx: RunContext, chat_cfg: Optional[ChatAgentConfig] = None) -> dict[str, Any]:
    """合并 memory（extra.l0_config）与显式传入的 chat 配置。"""
    merged: dict[str, Any] = dict(_extra(ctx).get("l0_config") or {})
    if chat_cfg is None:
        return merged
    merged.update(
        {
            "l0_context_compress_enabled": chat_cfg.l0_context_compress_enabled,
            "context_compress_threshold": chat_cfg.context_compress_threshold,
            "compress_target_ratio": chat_cfg.compress_target_ratio,
            "l0_context_window_tokens": chat_cfg.context_window_tokens,
        }
    )
    return merged


def _effective_compressor_config(
    ctx: RunContext,
    compressor: HermesContextCompressor,
    chat_cfg: Optional[ChatAgentConfig] = None,
) -> CompressorConfig:
    l0_cfg = resolve_l0_config(ctx, chat_cfg)
    if chat_cfg is not None or l0_cfg:
        return compressor_config_from_dict(l0_cfg)
    return compressor.config


def resolve_model_window_tokens(
    ctx: RunContext,
    chat_cfg: Optional[ChatAgentConfig] = None,
) -> int:
    """按 chat model_role 从 models.yml 解析 context window，否则回退 chat/memory 配置。"""
    if chat_cfg is None:
        from app.agents.orchestration.chat_config import load_chat_config

        chat_cfg = load_chat_config()
    models = getattr(ctx, "models", None)
    if models is not None:
        getter = getattr(models, "get_context_window_tokens", None)
        if callable(getter):
            window = getter(chat_cfg.model_role)
            if window and window > 0:
                return int(window)
    l0_cfg = resolve_l0_config(ctx, chat_cfg)
    for key in ("context_window_tokens", "l0_context_window_tokens"):
        raw = l0_cfg.get(key)
        if raw is not None:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return int(chat_cfg.context_window_tokens or 128000)


def is_l0_enabled(ctx: RunContext, chat_cfg: Optional[ChatAgentConfig] = None) -> bool:
    comp = get_context_compressor(ctx)
    if comp is None:
        return False
    cfg = _effective_compressor_config(ctx, comp, chat_cfg)
    return cfg.enabled


def clear_l0_state(ctx: RunContext) -> None:
    extra = _extra(ctx)
    extra.pop(_L0_STATE_KEY, None)


def get_l0_state(ctx: RunContext) -> Optional[dict[str, Any]]:
    state = _extra(ctx).get(_L0_STATE_KEY)
    return state if isinstance(state, dict) else None


def _split_summary_and_tail(messages: List[dict]) -> tuple[Optional[dict], List[dict]]:
    """从压缩结果中拆出 summary system 与 tail（不含 protected system 前缀）。"""
    if not messages:
        return None, []

    idx = 0
    while idx < len(messages) and messages[idx].get("role") == "system":
        content = str(messages[idx].get("content") or "")
        if _SUMMARY_MARKER in content:
            summary = messages[idx]
            return summary, list(messages[idx + 1 :])
        idx += 1

    # 无独立 summary：protected 之后全部视为 tail
    protected_end = 0
    while protected_end < len(messages) and messages[protected_end].get("role") == "system":
        protected_end += 1
    if protected_end < len(messages) and messages[protected_end].get("role") == "user":
        protected_end += 1
    return None, list(messages[protected_end:])


def apply_l0_to_turn_messages(
    ctx: RunContext,
    messages: List[dict],
    *,
    memory_snapshot_hash: str,
    chat_cfg: Optional[ChatAgentConfig] = None,
) -> List[dict]:
    """若存在有效 L0 状态，用 summary+tail 替换 L2 history 段（保留 fresh system）。"""
    if not is_l0_enabled(ctx, chat_cfg):
        return messages
    state = get_l0_state(ctx)
    if not state:
        return messages
    if state.get("memory_snapshot_hash") and state["memory_snapshot_hash"] != memory_snapshot_hash:
        clear_l0_state(ctx)
        return messages

    if not messages:
        return messages

    system_msg = messages[0]
    current_user = messages[-1] if messages[-1].get("role") == "user" else None
    if current_user is None:
        return messages

    summary_msg = state.get("summary_message")
    tail = state.get("tail_messages") or []
    rebuilt: List[dict] = [system_msg]
    if summary_msg:
        rebuilt.append(dict(summary_msg))
    rebuilt.extend(dict(m) for m in tail)
    rebuilt.append(dict(current_user))
    return rebuilt


def _store_l0_state(
    ctx: RunContext,
    compressed: List[dict],
    *,
    memory_snapshot_hash: str,
    result: Any,
) -> None:
    summary_msg, tail = _split_summary_and_tail(compressed)
    _extra(ctx)[_L0_STATE_KEY] = {
        "memory_snapshot_hash": memory_snapshot_hash,
        "summary_message": summary_msg,
        "tail_messages": tail,
        "original_token_count": getattr(result, "original_token_count", 0),
        "compressed_token_count": getattr(result, "compressed_token_count", 0),
        "savings_pct": getattr(result, "savings_pct", 0.0),
    }


def pop_last_prompt_tokens(ctx: RunContext) -> Optional[int]:
    extra = _extra(ctx)
    raw = extra.pop("_last_llm_prompt_tokens", None)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def store_prompt_tokens_from_response(ctx: RunContext, response: Any) -> None:
    """从 LLM 响应写入 prompt_tokens，供压缩阈值判断。"""
    usage = getattr(response, "usage", None)
    if usage is not None:
        raw = getattr(usage, "prompt_tokens", None)
        if raw is not None:
            _extra(ctx)["_last_llm_prompt_tokens"] = int(raw)
            return
    if isinstance(response, dict):
        raw = (response.get("usage") or {}).get("prompt_tokens")
        if raw is not None:
            _extra(ctx)["_last_llm_prompt_tokens"] = int(raw)


async def maybe_compress_turn_context(
    ctx: RunContext,
    messages: List[dict],
    *,
    chat_cfg: Optional[ChatAgentConfig] = None,
    memory_snapshot_hash: str = "",
    model_window: Optional[int] = None,
    prompt_tokens: Optional[int] = None,
) -> List[dict]:
    """LLM 回合结束后评估并压缩；不触发 L1 快照刷新。"""
    span_attrs: dict[str, Any] = {}
    async with span_ctx(ctx, "memory.l0.maybe_compress", Layer.AGENT, span_attrs):
        if not messages or not is_l0_enabled(ctx, chat_cfg):
            return messages

        compressor = get_context_compressor(ctx)
        if compressor is None:
            return messages

        cfg = _effective_compressor_config(ctx, compressor, chat_cfg)
        if model_window is None or model_window <= 0:
            if chat_cfg is not None or getattr(ctx, "models", None) is not None:
                model_window = resolve_model_window_tokens(ctx, chat_cfg)
            else:
                model_window = cfg.context_window_tokens
        window = compressor.resolve_model_window(model_window)
        usage_tokens = prompt_tokens if prompt_tokens is not None else pop_last_prompt_tokens(ctx)

        provider_context = ""
        from app.agents.memory.l4_context import get_memory_manager

        mgr = get_memory_manager(ctx)
        if mgr is not None:
            provider_context = await mgr.collect_pre_compress(messages, ctx.request)

        result = await compressor.maybe_compress(
            messages,
            ctx.request,
            window,
            prompt_tokens=usage_tokens,
            provider_context=provider_context,
        )

        if result.savings_pct > 0 and result.compressed_messages is not messages:
            record_l0_compress_triggered(ctx)
            span_attrs["compress_triggered"] = True
            span_attrs["savings_pct"] = round(result.savings_pct * 100, 2)
            _store_l0_state(
                ctx,
                result.compressed_messages,
                memory_snapshot_hash=memory_snapshot_hash,
                result=result,
            )
            extra = _extra(ctx)
            extra["_l0_compress_count"] = int(extra.get("_l0_compress_count") or 0) + 1
            if result.session_split:
                memory = ctx.memory
                split_fn = getattr(memory, "split_session_on_compression", None)
                if split_fn is not None:
                    new_session_id = f"{ctx.request.session_id}-c{int(result.compressed_token_count)}"
                    try:
                        split_result = await split_fn(ctx.request, new_session_id)
                        _extra(ctx)["_l0_session_split"] = split_result
                        mode = split_result.get("mode")
                        if mode == "inplace":
                            # inplace：同 session 继续，仅标记 continuation（不新建 L2 session）
                            _extra(ctx)["_l0_compression_continuation"] = "inplace"
                        elif mode == "split" and split_result.get("session_id"):
                            ctx.request = replace(
                                ctx.request,
                                session_id=str(split_result["session_id"]),
                            )
                    except Exception as exc:
                        _logger.warning("L2 compression session split failed: %s", exc)
            _logger.info(
                "L0 context stored for session=%s savings=%.1f%%",
                ctx.request.session_id,
                result.savings_pct * 100,
            )
            return result.compressed_messages

        return messages
