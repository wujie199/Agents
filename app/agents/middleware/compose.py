# -*- coding: utf-8 -*-
"""Middleware 组合：洋葱模型 wrap_node(middlewares, fn)。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from app.agents.middleware import Middleware


async def wrap_node(
    middlewares: List[Middleware],
    node_fn: Callable,
    node_name: str,
) -> Callable:
    """将 middleware 列表按洋葱模型包裹节点函数。

    执行顺序：
      middleware[0].on_enter → middleware[1].on_enter → ... → node_fn
      → ... → middleware[1].on_exit → middleware[0].on_exit

    Args:
        middlewares: 按顺序的 middleware 列表
        node_fn: 原始节点函数 async (state, config) -> dict
        node_name: 节点名称

    Returns:
        包裹后的节点函数
    """

    async def wrapped(state: Any, config: Any) -> Any:
        # 进入阶段
        extra: Dict[str, Any] = {}
        for mw in middlewares:
            ctx = await mw.on_enter(node_name, state, config)
            extra.update(ctx)

        # 执行节点
        error: Optional[Exception] = None
        result: Any = None
        try:
            result = await node_fn(state, config)
        except Exception as e:
            error = e

        # 退出阶段（逆序）
        for mw in reversed(middlewares):
            try:
                await mw.on_exit(
                    node_name, state, config, result,
                    error=error,
                    extra=extra,
                )
            except Exception:
                pass  # middleware 退出不应影响主流程

        if error is not None:
            raise error
        return result

    return wrapped


def compose_middlewares(
    *middlewares: Middleware,
) -> List[Middleware]:
    """按推荐顺序组合 middleware 列表。"""
    return list(middlewares)
