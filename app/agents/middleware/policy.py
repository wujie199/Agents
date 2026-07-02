# -*- coding: utf-8 -*-
"""PolicyMiddleware：ACL / 限流校验。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.agents.middleware.rate_limit_backend import QpsBackend, create_qps_backend

_logger = logging.getLogger("app.agents.middleware.policy")


class PolicyMiddleware:
    """在节点执行前校验 ACL 和限流策略。"""

    def __init__(
        self,
        *,
        max_qps_per_tenant: int = 100,
        max_parallel_sends: int = 5,
        rate_limit_backend: str = "memory",
        redis_url: Optional[str] = None,
        qps_backend: Optional[QpsBackend] = None,
    ) -> None:
        self._max_qps = max_qps_per_tenant
        self._max_parallel = max_parallel_sends
        self._qps = qps_backend or create_qps_backend(
            backend=rate_limit_backend,
            redis_url=redis_url,
        )

    @property
    def name(self) -> str:
        return "policy"

    async def on_enter(
        self,
        node_name: str,
        state: Any,
        config: Any,
    ) -> Dict[str, Any]:
        configurable = (config or {}).get("configurable") or {}
        ctx = configurable.get("run_ctx")
        tenant_id = getattr(getattr(ctx, "request", None), "tenant_id", None) if ctx else None

        if tenant_id and self._max_qps > 0:
            if not self._qps.allow(str(tenant_id), self._max_qps):
                raise PermissionError(
                    f"Tenant {tenant_id} exceeded QPS limit {self._max_qps}"
                )

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
        pass
