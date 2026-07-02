# -*- coding: utf-8 -*-
"""图级 Middleware 体系：洋葱模型包裹图节点。

执行顺序：
  RequestContextMiddleware → TracingMiddleware → TimingMiddleware → MetricsMiddleware
  → PolicyMiddleware → LoggingMiddleware → PrivacyMiddleware → ErrorClassifierMiddleware
  → [业务节点] → AuditMiddleware

使用方式：
  wrapped_node = wrap_node(middlewares, original_node_fn)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class Middleware(Protocol):
    """Middleware 协议。"""

    @property
    def name(self) -> str:
        ...

    async def on_enter(
        self,
        node_name: str,
        state: Any,
        config: Any,
    ) -> Dict[str, Any]:
        """节点执行前。返回 extra context 注入后续 middleware。"""
        ...

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
        """节点执行后。"""
        ...


__all__ = ["Middleware", "wrap_node", "compose_middlewares"]
