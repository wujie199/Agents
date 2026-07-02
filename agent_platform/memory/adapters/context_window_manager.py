# -*- coding: utf-8 -*-
"""L0 上下文窗口分区：Zone 1 静态 system、Zone 2 冻结 L1、Zone 3 可变 history。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple


def _message_tokens(msg: dict, estimate_fn) -> int:
    content = msg.get("content", "") or ""
    tokens = estimate_fn(str(content))
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        tokens += estimate_fn(str(fn.get("name", "")))
        tokens += estimate_fn(str(fn.get("arguments", "")))
    return tokens


@dataclass(frozen=True)
class ContextZones:
    """消息列表的三段分区。"""

    protected_head: List[dict]  # Zone 1+2：system 前缀 + 首条 user
    middle: List[dict]          # 可压缩段
    tail: List[dict]            # 尾部保留预算内消息


class ContextWindowManager:
    """按 Hermes 规则划分上下文窗口，压缩时保护前缀。"""

    def __init__(self, estimate_tokens):
        self._estimate = estimate_tokens

    def split_zones(
        self,
        messages: Sequence[dict],
        *,
        tail_token_budget: int,
    ) -> ContextZones:
        if not messages:
            return ContextZones([], [], [])

        protected_end = self._protected_head_end(messages)
        protected = list(messages[:protected_end])
        rest = list(messages[protected_end:])

        if not rest or tail_token_budget <= 0:
            return ContextZones(protected, rest, [])

        tail: List[dict] = []
        used = 0
        for msg in reversed(rest):
            msg_tokens = _message_tokens(msg, self._estimate)
            if tail and used + msg_tokens > tail_token_budget:
                break
            tail.insert(0, msg)
            used += msg_tokens

        middle = rest[: len(rest) - len(tail)] if tail else rest
        return ContextZones(protected, middle, tail)

    @staticmethod
    def _protected_head_end(messages: Sequence[dict]) -> int:
        """锚定保护：所有 leading system + 首条 user。"""
        idx = 0
        while idx < len(messages) and messages[idx].get("role") == "system":
            idx += 1
        if idx < len(messages) and messages[idx].get("role") == "user":
            idx += 1
        return idx

    def reassemble(
        self,
        protected: Sequence[dict],
        summary_message: dict | None,
        tail: Sequence[dict],
    ) -> List[dict]:
        out: List[dict] = list(protected)
        if summary_message and (summary_message.get("content") or "").strip():
            out.append(summary_message)
        out.extend(tail)
        return out

    def tail_start_index(self, messages: Sequence[dict], zones: ContextZones) -> int:
        return len(zones.protected_head) + len(zones.middle)

    def should_compress(
        self,
        prompt_tokens: int,
        model_window: int,
        *,
        trigger_pct: float = 0.50,
    ) -> bool:
        if model_window <= 0 or prompt_tokens <= 0:
            return False
        return prompt_tokens >= int(model_window * trigger_pct)

    def update_from_response(
        self,
        messages: List[dict],
        usage: dict[str, Any] | None,
        *,
        model_window: int,
        trigger_pct: float = 0.50,
    ) -> tuple[List[dict], bool]:
        prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
        return messages, self.should_compress(
            prompt_tokens, model_window, trigger_pct=trigger_pct
        )


def prune_old_tool_results(
    messages: List[dict],
    *,
    tail_start: int,
    min_chars: int = 200,
    placeholder: str = "[tool result truncated]",
) -> Tuple[List[dict], int]:
    """Phase 1 prune：tail 之外的 tool 结果超长则替换为占位符。"""
    pruned = 0
    out: List[dict] = []
    for i, msg in enumerate(messages):
        if i >= tail_start and msg.get("role") == "tool":
            out.append(dict(msg))
            continue
        if msg.get("role") != "tool":
            out.append(dict(msg))
            continue
        content = str(msg.get("content") or "")
        if len(content) > min_chars:
            copy = dict(msg)
            copy["content"] = placeholder
            out.append(copy)
            pruned += 1
        else:
            out.append(dict(msg))
    return out, pruned


def repair_tool_message_pairs(messages: List[dict]) -> List[dict]:
    """修复压缩后孤立的 tool_call / tool_result 对。"""
    if not messages:
        return []

    repaired: List[dict] = []
    i = 0
    while i < len(messages):
        msg = dict(messages[i])
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            tool_calls = msg.get("tool_calls") or []
            call_ids = [
                str(tc.get("id") or "")
                for tc in tool_calls
                if isinstance(tc, dict) and tc.get("id")
            ]
            repaired.append(msg)
            i += 1
            if not call_ids:
                continue

            seen: set[str] = set()
            while i < len(messages) and messages[i].get("role") == "tool":
                tool_msg = dict(messages[i])
                tcid = str(tool_msg.get("tool_call_id") or "")
                if tcid in call_ids:
                    repaired.append(tool_msg)
                    seen.add(tcid)
                i += 1

            for missing in set(call_ids) - seen:
                repaired.append(
                    {
                        "role": "tool",
                        "tool_call_id": missing,
                        "content": "[tool result missing after compression]",
                    }
                )
            continue

        if role == "tool":
            i += 1
            continue

        repaired.append(msg)
        i += 1

    return repaired
