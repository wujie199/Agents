# -*- coding: utf-8 -*-
"""Hermes 风格的 L1 记忆安全扫描：威胁 regex + 不可见 Unicode 检测。"""

from __future__ import annotations

import re
from typing import Optional

# ── 12 条威胁 regex（从 Hermes 移植） ──
_THREAT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(previous|all|above)\s+instructions?", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"system\s*:\s*", re.I),
    re.compile(r"<\|im_start\|>", re.I),
    re.compile(r"###\s*Instruction", re.I),
    re.compile(r"pretend\s+you\s+are", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"inject\s+prompt", re.I),
    re.compile(r"override\s+system", re.I),
    re.compile(r"new\s+rules?\s*:", re.I),
    re.compile(r"forget\s+(everything|all|previous)", re.I),
    re.compile(r"act\s+as\s+if", re.I),
]

# ── 10 种不可见 Unicode 范围（从 Hermes 移植） ──
_INVISIBLE_UNICODE_RANGES: list[tuple[int, int]] = [
    (0x200B, 0x200F),  # ZWSP, ZWNJ, ZWJ, LRM, RLM
    (0x2028, 0x202F),  # line/para sep, control chars
    (0x2060, 0x206F),  # word joiner, invisible operators
    (0xFE00, 0xFE0F),  # VS1-VS16
    (0xFFF9, 0xFFFB),  # interlinear annotation
    (0xE0000, 0xE007F),  # tags
]


def scan_memory_content(content: str) -> Optional[str]:
    """扫描记忆内容，返回威胁描述或 None。

    多租户适配：纯函数无状态，调用方在日志中记录 tenant_id 以便审计追踪。
    """
    if not content:
        return None

    # 1. 威胁 regex
    for pattern in _THREAT_PATTERNS:
        if pattern.search(content):
            return f"Threat pattern matched: {pattern.pattern}"

    # 2. 不可见 Unicode
    for char in content:
        cp = ord(char)
        for lo, hi in _INVISIBLE_UNICODE_RANGES:
            if lo <= cp <= hi:
                return f"Invisible Unicode U+{cp:04X} detected"

    return None
