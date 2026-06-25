# -*- coding: utf-8 -*-
"""时间衰减：session_search 检索结果按时间距离降权。

decay_factor = 0.5 ^ (days_since / half_life)
半年前的结果权重约为 5 分钟前结果的 1/4（half_life=90 天时）。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Optional


def time_decay_factor(
    ts: str,
    *,
    now: Optional[datetime] = None,
    half_life_days: float = 90.0,
) -> float:
    """计算时间衰减因子。

    Args:
        ts: ISO 8601 时间戳字符串
        now: 当前时间（可注入，默认 UTC now）
        half_life_days: 半衰期天数（默认 90 天）

    Returns:
        衰减因子 0.0-1.0（越近越大）
    """
    if not ts or not ts.strip():
        return 1.0
    try:
        event_time = _parse_iso(ts)
    except (ValueError, TypeError):
        return 1.0

    ref = now or datetime.now(timezone.utc)
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)

    delta = (ref - event_time).total_seconds()
    if delta <= 0:
        return 1.0  # 未来时间不衰减

    days_since = delta / 86400.0
    if half_life_days <= 0:
        return 1.0

    return 0.5 ** (days_since / half_life_days)


def apply_time_decay_to_fragments(
    fragments: List[dict],
    *,
    now: Optional[datetime] = None,
    half_life_days: float = 90.0,
    ts_field: str = "ts",
    score_field: str = "score",
) -> List[dict]:
    """对检索结果列表应用时间衰减，重新排序。

    Args:
        fragments: 检索结果列表，每项包含 ts 和 score 字段
        now: 当前时间
        half_life_days: 半衰期天数
        ts_field: 时间戳字段名
        score_field: 分数字段名

    Returns:
        按衰减后 score 降序排列的列表
    """
    if not fragments:
        return []

    result = []
    for frag in fragments:
        ts = str(frag.get(ts_field) or "")
        original_score = float(frag.get(score_field) or 0.0)
        decay = time_decay_factor(ts, now=now, half_life_days=half_life_days)
        decayed_score = original_score * decay
        result.append({
            **frag,
            score_field: decayed_score,
            "_decay_factor": round(decay, 4),
            "_original_score": original_score,
        })

    result.sort(key=lambda x: x.get(score_field, 0.0), reverse=True)
    return result


def _parse_iso(ts: str) -> datetime:
    """解析 ISO 8601 时间戳（容错）。"""
    ts = ts.strip()
    # 尝试标准解析
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    # 回退：fromisoformat
    return datetime.fromisoformat(ts)
