# -*- coding: utf-8 -*-
"""模型输出清洗：移除 reasoning / thinking 块，避免污染 L2 与 prompt。"""

from __future__ import annotations

import re

_THINKING_BLOCK = re.compile(
    r"<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)
_THINKING_TAIL = re.compile(
    r"<think>.*",
    re.DOTALL | re.IGNORECASE,
)
_THINKING_OPEN = re.compile(r"<think>", re.IGNORECASE)


def strip_model_reasoning(text: str | None) -> str:
    """移除 `` 及未闭合片段。"""
    if not text:
        return ""
    cleaned = _THINKING_BLOCK.sub("", text)
    cleaned = re.sub(
        r"<think>.*?(?:\n\n+|$)",
        "\n",
        cleaned,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = _THINKING_TAIL.sub("", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def has_model_reasoning(text: str | None) -> bool:
    """是否仍含 thinking / reasoning 标记。"""
    if not text:
        return False
    return bool(_THINKING_OPEN.search(text))


def sanitize_turn_content(text: str | None, *, role: str = "") -> str:
    """历史消息清洗：assistant 去 thinking 与 JSON 代码块，user 原样。"""
    raw = (text or "").strip()
    if role == "assistant":
        return sanitize_search_fragment_content(raw, role="assistant")
    return raw


def sanitize_search_fragment_content(text: str | None, *, role: str = "") -> str:
    """L2 session_search 片段：移除 thinking，assistant 额外去 JSON 代码块包裹。"""
    cleaned = strip_model_reasoning(text)
    if role == "assistant":
        cleaned = re.sub(r"```(?:json)?\s*[\s\S]*?```", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


_JSON_OUTPUT_FORMAT = re.compile(r"^输出格式:\s*(JSON|json)\s*$", re.MULTILINE)


def sanitize_memory_text_for_chat(text: str | None) -> str:
    """L1 注入 prompt 时：丢弃误抽取的 JSON 输出格式，默认自然语言。"""
    raw = (text or "").strip()
    if not raw:
        return "【回复风格】默认使用自然语言；仅当用户明确要求 JSON/结构化输出时再使用 JSON。"
    body = _JSON_OUTPUT_FORMAT.sub("", raw)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    style = (
        "【回复风格】默认使用自然语言；"
        "仅当用户明确要求 JSON/结构化输出时再使用 JSON。"
    )
    return f"{body}\n\n{style}" if body else style
