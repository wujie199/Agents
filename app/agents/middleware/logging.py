# -*- coding: utf-8 -*-
"""LoggingMiddleware：节点开始/结束/耗时日志。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

_logger = logging.getLogger("app.agents.middleware")


class LoggingMiddleware:
    """记录每个图节点的进入、退出和耗时。"""

    @property
    def name(self) -> str:
        return "logging"

    async def on_enter(
        self,
        node_name: str,
        state: Any,
        config: Any,
    ) -> Dict[str, Any]:
        _logger.info("node=%s action=enter", node_name)
        return {}

    async def on_exit(
        self,
        node_name: str,
        state: Any,
        config: Any,
        result: Any,
        *,
        error: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = extra or {}
        duration_ms = ctx.get("duration_ms")
        if error:
            _logger.error(
                "node=%s action=exit error=%s duration_ms=%s",
                node_name,
                error,
                duration_ms,
            )
        else:
            _logger.info(
                "node=%s action=exit duration_ms=%s",
                node_name,
                duration_ms,
            )
