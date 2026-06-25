# -*- coding: utf-8 -*-
"""L1 记忆冲突检测：同 key 新旧 value 矛盾时策略处理。

策略：
- overwrite: 直接覆盖（默认行为）
- keep_old: 保留旧值，忽略新值
- ask_user: 标记冲突，等待 HITL 确认
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

ConflictStrategy = Literal["overwrite", "keep_old", "ask_user"]


@dataclass(frozen=True)
class ConflictRecord:
    """冲突记录。"""

    key: str
    old_value: str
    new_value: str
    strategy: ConflictStrategy
    resolved_value: str  # 最终写入的值
    needs_hitl: bool = False  # 是否需要人工确认


# 明确事实的高置信模式（姓名、称呼等）
_HIGH_CONFIDENCE_PATTERNS = [
    re.compile(r"^(我叫|我的名字是|名字是|我是)\s*[\u4e00-\u9fa5a-zA-Z]{1,20}$", re.I),
    re.compile(r"^(我的称呼是|叫我)\s*.+$", re.I),
]

# 值等价判断：去除空格/标点后比较
_NORMALIZE_RE = re.compile(r"[\s\u3000,，。.、！!？?]+")


def _normalize_value(value: str) -> str:
    """归一化值用于等价比较。"""
    return _NORMALIZE_RE.sub("", (value or "").strip().lower())


def is_values_conflicting(old_value: str, new_value: str) -> bool:
    """判断新旧值是否冲突。

    同义不冲突（如 "张三" vs "我叫张三"），
    矛盾才冲突（如 "张三" vs "李四"）。
    """
    old_norm = _normalize_value(old_value)
    new_norm = _normalize_value(new_value)
    if not old_norm or not new_norm:
        return False
    # 完全等价
    if old_norm == new_norm:
        return False
    # 新值包含旧值（如 "张三" in "我叫张三"）
    if old_norm in new_norm or new_norm in old_norm:
        return False
    # 视为冲突
    return True


def compute_confidence(key: str, value: str, context: str = "") -> float:
    """计算 L1 事实置信度。

    Returns:
        0.0-1.0，越高越可信
    """
    # 明确事实高置信
    full = f"{key}: {value}" if key else value
    for pattern in _HIGH_CONFIDENCE_PATTERNS:
        if pattern.match(full.strip()):
            return 0.95

    # 常见明确 key
    high_conf_keys = {"姓名", "名字", "称呼", "语言", "职业", "时区"}
    if key in high_conf_keys:
        return 0.90

    # 通用事实
    medium_conf_keys = {"项目", "输出格式", "偏好"}
    if key in medium_conf_keys:
        return 0.75

    # 默认中等置信
    return 0.6


def resolve_conflict(
    key: str,
    old_value: str,
    new_value: str,
    strategy: ConflictStrategy = "ask_user",
    *,
    new_confidence: Optional[float] = None,
    l1_auto_write_confidence_min: float = 0.9,
) -> ConflictRecord:
    """解决记忆冲突。

    Args:
        key: L1 key
        old_value: 当前存储的值
        new_value: 新写入的值
        strategy: 冲突策略
        new_confidence: 新值的置信度
        l1_auto_write_confidence_min: 自动写入的最低置信度

    Returns:
        ConflictRecord: 冲突解决结果
    """
    if not is_values_conflicting(old_value, new_value):
        # 不冲突，直接覆盖
        return ConflictRecord(
            key=key,
            old_value=old_value,
            new_value=new_value,
            strategy="overwrite",
            resolved_value=new_value,
            needs_hitl=False,
        )

    confidence = new_confidence if new_confidence is not None else compute_confidence(key, new_value)

    if strategy == "overwrite":
        return ConflictRecord(
            key=key,
            old_value=old_value,
            new_value=new_value,
            strategy="overwrite",
            resolved_value=new_value,
            needs_hitl=False,
        )
    elif strategy == "keep_old":
        return ConflictRecord(
            key=key,
            old_value=old_value,
            new_value=new_value,
            strategy="keep_old",
            resolved_value=old_value,
            needs_hitl=False,
        )
    elif strategy == "ask_user":
        # 高置信自动覆盖，低置信走 HITL
        if confidence >= l1_auto_write_confidence_min:
            return ConflictRecord(
                key=key,
                old_value=old_value,
                new_value=new_value,
                strategy="ask_user",
                resolved_value=new_value,
                needs_hitl=True,  # 仍需确认但预选新值
            )
        return ConflictRecord(
            key=key,
            old_value=old_value,
            new_value=new_value,
            strategy="ask_user",
            resolved_value=old_value,  # 保留旧值，待用户确认
            needs_hitl=True,
        )

    # 默认 fallback
    return ConflictRecord(
        key=key,
        old_value=old_value,
        new_value=new_value,
        strategy=strategy,
        resolved_value=new_value,
        needs_hitl=False,
    )


def check_l1_write_conflicts(
    existing_facts: Dict[str, str],
    new_deltas: List[Dict[str, str]],
    strategy: ConflictStrategy = "ask_user",
    *,
    l1_auto_write_confidence_min: float = 0.9,
) -> List[ConflictRecord]:
    """批量检查 L1 写入冲突。

    Args:
        existing_facts: 当前 L1 KV（key → value）
        new_deltas: 待写入的 deltas（每项含 key, value）
        strategy: 冲突策略
        l1_auto_write_confidence_min: 自动写入最低置信度

    Returns:
        冲突记录列表
    """
    records: List[ConflictRecord] = []
    for delta in new_deltas:
        key = str(delta.get("key") or "").strip()
        new_value = str(delta.get("value") or "").strip()
        if not key or not new_value:
            continue
        old_value = existing_facts.get(key, "")
        if not old_value:
            # 无旧值，不冲突
            confidence = compute_confidence(key, new_value)
            records.append(ConflictRecord(
                key=key,
                old_value="",
                new_value=new_value,
                strategy="overwrite",
                resolved_value=new_value,
                needs_hitl=confidence < l1_auto_write_confidence_min,
            ))
            continue
        record = resolve_conflict(
            key,
            old_value,
            new_value,
            strategy,
            new_confidence=compute_confidence(key, new_value),
            l1_auto_write_confidence_min=l1_auto_write_confidence_min,
        )
        records.append(record)

    conflicts = [r for r in records if is_values_conflicting(r.old_value, r.new_value)]
    if conflicts:
        logger.info(
            "L1 conflict detected: %d conflicts out of %d deltas",
            len(conflicts),
            len(new_deltas),
        )

    return records
