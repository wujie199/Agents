# -*- coding: utf-8 -*-
"""AuditMiddleware：关键字段 hash 审计日志。"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.agents.middleware.audit_persist import persist_audit_event

_logger = logging.getLogger("app.agents.middleware.audit")


class AuditMiddleware:
    """节点执行后记录关键字段的 hash 审计条目。"""

    def __init__(
        self,
        *,
        persist: bool = False,
        audit_log_dir: str = "data/audit",
    ) -> None:
        self._persist = persist
        self._audit_log_dir = audit_log_dir

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

        configurable = (config or {}).get("configurable") or {}
        run_ctx = configurable.get("run_ctx")
        tenant_id = user_id = session_id = ""
        if run_ctx is not None:
            req = getattr(run_ctx, "request", None)
            if req is not None:
                tenant_id = str(getattr(req, "tenant_id", "") or "")
                user_id = str(getattr(req, "user_id", "") or "")
                session_id = str(getattr(req, "session_id", "") or "")

        content_hashes: Dict[str, str] = {}
        if isinstance(result, dict):
            for key in ("assistant_text", "user_input"):
                value = result.get(key)
                if value and isinstance(value, str):
                    content_hashes[key] = _sha256(value)

        audit_entry = {
            "trace_id": trace_id,
            "span_id": span_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "session_id": session_id,
            "node": node_name,
            "duration_ms": duration_ms,
            "error": str(error) if error else None,
            "content_hashes": content_hashes,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        _logger.info("audit %s", audit_entry)
        if run_ctx is not None and isinstance(getattr(run_ctx, "extra", None), dict):
            turn_audit = run_ctx.extra.setdefault("turn_audit", [])
            if isinstance(turn_audit, list):
                turn_audit.append(dict(audit_entry))
        if self._persist:
            persist_audit_event(audit_entry, self._audit_log_dir)


def _sha256(text: str) -> str:
    """计算 SHA-256 hash（前 16 字符）。"""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
