# -*- coding: utf-8 -*-
"""RequestContextMiddleware：trace_id 解析与 RunContext 身份注入。"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional


class RequestContextMiddleware:
    """解析 trace_id，将 tenant/session/user 写入 span 属性上下文。"""

    def __init__(self, *, trace_header: str = "X-Request-ID") -> None:
        self._trace_header = trace_header

    @property
    def name(self) -> str:
        return "request_context"

    async def on_enter(
        self,
        node_name: str,
        state: Any,
        config: Any,
    ) -> Dict[str, Any]:
        configurable = (config or {}).get("configurable") or {}
        run_ctx = configurable.get("run_ctx")

        trace_id = configurable.get("trace_id")
        if not trace_id and run_ctx is not None:
            req = getattr(run_ctx, "request", None)
            if req is not None:
                trace_id = getattr(req, "trace_id", None)
        if not trace_id:
            trace_id = str(uuid.uuid4())

        if isinstance(config, dict):
            cfg_block = config.setdefault("configurable", {})
            cfg_block["trace_id"] = trace_id

        span_attributes: Dict[str, str] = {"node": node_name}
        if run_ctx is not None:
            req = getattr(run_ctx, "request", None)
            if req is not None:
                for key in ("tenant_id", "session_id", "user_id"):
                    val = getattr(req, key, None)
                    if val:
                        span_attributes[key] = str(val)

        if isinstance(config, dict):
            config["configurable"]["_span_attributes"] = span_attributes

        return {
            "trace_id": trace_id,
            "span_attributes": span_attributes,
        }

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
        return
