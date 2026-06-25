# -*- coding: utf-8 -*-
"""AuditMiddleware：关键字段 hash 审计日志。"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Optional

_logger = logging.getLogger("app.agents.middleware.audit")


class AuditMiddleware:
    """节点执行后记录关键字段的 hash 审计条目。"""

    @property
    def name(self) -> str:
        return "audit"

    async def on_enter(
        self,
        node_name: str,
        state: Any,
        config: Any,
    ) -> Dict[str, Any]:
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
        trace_id = ctx.get("trace_id", "unknown")
        span_id = ctx.get("span_id", "unknown")
        duration_ms = ctx.get("duration_ms")

        audit_entry = {
            "trace_id": trace_id,
            "span_id": span_id,
            "node": node_name,
            "duration_ms": duration_ms,
            "error": str(error) if error else None,
        }

        # 对 result 中的敏感字段做 hash
        if isinstance(result, dict):
            for key in ("assistant_text", "user_input"):
                value = result.get(key)
                if value and isinstance(value, str):
                    audit_entry[f"{key}_hash"] = _sha256(value)

        _logger.info("audit %s", audit_entry)


def _sha256(text: str) -> str:
    """计算 SHA-256 hash（前 16 字符）。"""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
