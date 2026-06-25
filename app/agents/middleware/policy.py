# -*- coding: utf-8 -*-
"""PolicyMiddleware：ACL / 限流校验。"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

_logger = logging.getLogger("app.agents.middleware.policy")


class PolicyMiddleware:
    """在节点执行前校验 ACL 和限流策略。"""

    def __init__(
        self,
        *,
        max_qps_per_tenant: int = 100,
        max_parallel_sends: int = 5,
    ) -> None:
        self._max_qps = max_qps_per_tenant
        self._max_parallel = max_parallel_sends
        self._tenant_calls: Dict[str, list] = {}

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
            now = time.time()
            calls = self._tenant_calls.setdefault(tenant_id, [])
            # 清理 1 秒前的记录
            calls[:] = [t for t in calls if now - t < 1.0]
            if len(calls) >= self._max_qps:
                raise PermissionError(
                    f"Tenant {tenant_id} exceeded QPS limit {self._max_qps}"
                )
            calls.append(now)

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
