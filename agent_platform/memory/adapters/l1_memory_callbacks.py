# -*- coding: utf-8 -*-
"""L3/L4 记忆镜像钩子 — Phase D 接入 MemoryManager。"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

OnMemoryWriteCallback = Callable[[str, str, str, dict[str, Any]], None]


class NoOpMemoryWriteMirror:
    """默认 no-op：记录 debug 日志，不写入外部 provider。"""

    def __call__(
        self, tenant_id: str, user_id: str, action: str, payload: dict[str, Any]
    ) -> None:
        logger.debug(
            "on_memory_write stub tenant=%s user=%s action=%s target=%s",
            tenant_id,
            user_id,
            action,
            payload.get("target"),
        )


def build_on_memory_write_callback(
    external_memory: Any = None,
    memory_manager: Any = None,
) -> OnMemoryWriteCallback:
    """工厂：优先 MemoryManager.notify_memory_tool_write。"""
    if memory_manager is not None:
        from agent_platform.memory.adapters.memory_manager import (
            build_on_memory_write_from_manager,
        )

        return build_on_memory_write_from_manager(memory_manager)
    return NoOpMemoryWriteMirror()
