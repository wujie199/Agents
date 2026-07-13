# -*- coding: utf-8
"""Memory evaluation metrics."""

from __future__ import annotations


def keyword_recall(summary: str, keywords: list[str]) -> float:
    """期望关键词在 session_search 摘要中的命中率。"""
    if not keywords:
        return 1.0
    text = (summary or "").lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return hits / len(keywords)


def kv_match_score(extracted: dict[str, str], expected: dict[str, str]) -> float:
    """L1 抽取 KV 与 golden 的匹配率（key 存在且 value 包含或相等）。"""
    if not expected:
        return 1.0
    hits = 0
    for key, exp_val in expected.items():
        got = (extracted.get(key) or "").strip()
        exp = exp_val.strip()
        if got == exp or (exp and exp in got):
            hits += 1
    return hits / len(expected)


def user_snapshot_kv(snapshot_text: str) -> dict[str, str]:
    """从 compose_prompt_snapshot 文本解析 USER KV。"""
    out: dict[str, str] = {}
    in_user = False
    for line in (snapshot_text or "").splitlines():
        if line.strip().startswith("# USER"):
            in_user = True
            continue
        if in_user and line.startswith("# "):
            break
        if not in_user:
            continue
        if ": " in line.strip():
            k, v = line.strip().split(": ", 1)
            if k.strip():
                out[k.strip()] = v.strip()
    return out


def user_file_kv(user_raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (user_raw or "").splitlines():
        if ": " in line.strip():
            k, v = line.strip().split(": ", 1)
            if k.strip() and not k.startswith("#"):
                out[k.strip()] = v.strip()
    return out
