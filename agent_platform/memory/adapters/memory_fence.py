# -*- coding: utf-8 -*-
"""Hermes 风格的围栏封装与流式剥离。

多租户适配：围栏中携带 tenant={tenant_id} 标记，
剥离逻辑是纯文本处理，无需租户感知。
"""

from __future__ import annotations

import re
from typing import AsyncIterator


_FENCE_OPEN = "<memory-context>"
_FENCE_CLOSE = "</memory-context>"

_FENCE_RE = re.compile(
    rf"{re.escape(_FENCE_OPEN)}.*?{re.escape(_FENCE_CLOSE)}",
    re.DOTALL,
)


def build_memory_context_block(
    content: str, source: str, tenant_id: str
) -> str:
    """用围栏标记包裹外部记忆内容。"""
    return (
        f"{_FENCE_OPEN} source={source} tenant={tenant_id}\n"
        f"{content}\n"
        f"{_FENCE_CLOSE}"
    )


def strip_memory_fences(text: str) -> str:
    """从文本中剥离围栏标记（用户不可见内部封装）。"""
    return _FENCE_RE.sub("", text)


async def strip_memory_fences_from_stream(
    stream: AsyncIterator[str],
) -> AsyncIterator[str]:
    """流式输出时实时剥离围栏标记。"""
    buffer = ""
    async for chunk in stream:
        buffer += chunk
        # 尝试剥离已完成的围栏块
        buffer = _FENCE_RE.sub("", buffer)
        # 如果 buffer 中包含不完整的围栏开头，暂时保留
        if _FENCE_OPEN in buffer and _FENCE_CLOSE not in buffer:
            # 不完整围栏，等待更多数据
            open_pos = buffer.index(_FENCE_OPEN)
            safe_part = buffer[:open_pos]
            buffer = buffer[open_pos:]
            if safe_part:
                yield safe_part
        else:
            if buffer:
                yield buffer
            buffer = ""
    # 刷出残留
    if buffer:
        yield buffer
