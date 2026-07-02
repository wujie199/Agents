# -*- coding: utf-8 -*-
"""PolicyMiddleware 限流后端：内存滑动窗口 + 可选 Redis（fail-open）。"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Protocol

_logger = logging.getLogger("app.agents.middleware.rate_limit_backend")


class QpsBackend(Protocol):
    def allow(self, tenant_id: str, max_qps: int) -> bool: ...


class InMemoryQpsBackend:
    """进程内 1 秒滑动窗口。"""

    def __init__(self) -> None:
        self._tenant_calls: Dict[str, list[float]] = {}

    def allow(self, tenant_id: str, max_qps: int) -> bool:
        if max_qps <= 0:
            return True
        now = time.time()
        calls = self._tenant_calls.setdefault(tenant_id, [])
        calls[:] = [t for t in calls if now - t < 1.0]
        if len(calls) >= max_qps:
            return False
        calls.append(now)
        return True


class RedisQpsBackend:
    """Redis 1 秒滑动窗口（sorted set）；失败时回退内存。"""

    def __init__(
        self,
        redis_client: Any,
        *,
        fallback: Optional[InMemoryQpsBackend] = None,
        key_prefix: str = "obs:policy:qps",
    ) -> None:
        self._redis = redis_client
        self._fallback = fallback or InMemoryQpsBackend()
        self._key_prefix = key_prefix

    def allow(self, tenant_id: str, max_qps: int) -> bool:
        if max_qps <= 0:
            return True
        try:
            key = f"{self._key_prefix}:{tenant_id}"
            now = time.time()
            member = f"{now}:{time.monotonic_ns()}"
            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(key, 0, now - 1.0)
            pipe.zadd(key, {member: now})
            pipe.zcard(key)
            pipe.expire(key, 2)
            _, _, count, _ = pipe.execute()
            return int(count) <= max_qps
        except Exception as exc:
            _logger.debug("Redis QPS backend fail-open: %s", exc)
            return self._fallback.allow(tenant_id, max_qps)


def resolve_redis_url(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit.strip() or None
    return (os.environ.get("REDIS_URL") or "").strip() or None


def create_qps_backend(
    *,
    backend: str = "memory",
    redis_url: Optional[str] = None,
) -> QpsBackend:
    """按配置 / 环境变量选择限流后端；Redis 不可用时 fail-open 到内存。"""
    use_redis = backend == "redis" or (
        backend != "memory" and resolve_redis_url(redis_url) is not None
    )
    memory = InMemoryQpsBackend()
    url = resolve_redis_url(redis_url)
    if not use_redis or not url:
        return memory
    try:
        import redis

        client = redis.from_url(url, decode_responses=False)
        client.ping()
        return RedisQpsBackend(client, fallback=memory)
    except Exception as exc:
        _logger.warning("Policy Redis 限流不可用，回退内存: %s", exc)
        return memory
