# -*- coding: utf-8 -*-
"""聊天 API 限流：进程内滑动窗口 + 可选 Redis 固定窗口。"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from urllib.parse import urlparse, urlunparse
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Deque, Dict, Optional, Protocol, Tuple

import yaml


class RateLimiter(Protocol):
    def check(self, tenant_id: str, user_id: str) -> Tuple[bool, Optional[str]]: ...


@dataclass(frozen=True)
class ChatRateLimitConfig:
    tenant_requests_per_minute: int = 60
    user_requests_per_minute: int = 20
    enabled: bool = True


def load_chat_rate_limit_config(
    config_dir: str = "config",
) -> ChatRateLimitConfig:
    path = Path(config_dir) / "concurrency.yml"
    if not path.is_file():
        return ChatRateLimitConfig()
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    chat = raw.get("chat") or {}
    default = raw.get("default") or {}
    return ChatRateLimitConfig(
        enabled=bool(chat.get("enabled", True)),
        tenant_requests_per_minute=int(
            chat.get(
                "tenant_requests_per_minute",
                default.get("chat_tenant_requests_per_minute", 60),
            )
        ),
        user_requests_per_minute=int(
            chat.get(
                "user_requests_per_minute",
                default.get("chat_user_requests_per_minute", 20),
            )
        ),
    )


class SlidingWindowRateLimiter:
    """进程内滑动窗口限流（tenant + user 双维度）。"""

    def __init__(self, cfg: ChatRateLimitConfig) -> None:
        self._cfg = cfg
        self._tenant_hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._user_hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def _prune(self, hits: Deque[float], now: float, window: float = 60.0) -> None:
        cutoff = now - window
        while hits and hits[0] <= cutoff:
            hits.popleft()

    def check(self, tenant_id: str, user_id: str) -> Tuple[bool, Optional[str]]:
        if not self._cfg.enabled:
            return True, None
        now = time.monotonic()
        tenant_key = tenant_id or "default"
        user_key = f"{tenant_id}:{user_id}"

        with self._lock:
            tenant_hits = self._tenant_hits[tenant_key]
            user_hits = self._user_hits[user_key]
            self._prune(tenant_hits, now)
            self._prune(user_hits, now)

            if len(tenant_hits) >= self._cfg.tenant_requests_per_minute:
                return False, "tenant 请求过于频繁，请稍后再试"
            if len(user_hits) >= self._cfg.user_requests_per_minute:
                return False, "user 请求过于频繁，请稍后再试"

            tenant_hits.append(now)
            user_hits.append(now)
        return True, None


class RedisRateLimiter:
    """Redis 固定窗口限流（多实例共享）；失败时回退内存限流。"""

    def __init__(
        self,
        cfg: ChatRateLimitConfig,
        redis_client: Any,
        *,
        fallback: Optional[SlidingWindowRateLimiter] = None,
    ) -> None:
        self._cfg = cfg
        self._redis = redis_client
        self._fallback = fallback or SlidingWindowRateLimiter(cfg)

    def _incr_window(self, key: str, limit: int, window_sec: int = 60) -> bool:
        count = int(self._redis.incr(key))
        if count == 1:
            self._redis.expire(key, window_sec * 2)
        return count <= limit

    def check(self, tenant_id: str, user_id: str) -> Tuple[bool, Optional[str]]:
        if not self._cfg.enabled:
            return True, None
        try:
            window = int(time.time() // 60)
            tenant_key = f"chat:rl:tenant:{tenant_id or 'default'}:{window}"
            user_key = f"chat:rl:user:{tenant_id}:{user_id}:{window}"
            if not self._incr_window(
                tenant_key, self._cfg.tenant_requests_per_minute
            ):
                return False, "tenant 请求过于频繁，请稍后再试"
            if not self._incr_window(
                user_key, self._cfg.user_requests_per_minute
            ):
                return False, "user 请求过于频繁，请稍后再试"
            return True, None
        except Exception:
            return self._fallback.check(tenant_id, user_id)


def _resolve_rate_limit_redis_url() -> Optional[str]:
    """限流 Redis：优先 CHAT_RATE_LIMIT_REDIS_URL；否则 REDIS_URL 的 db/1。"""
    explicit = os.environ.get("CHAT_RATE_LIMIT_REDIS_URL")
    if explicit:
        return explicit
    base = os.environ.get("REDIS_URL")
    if not base:
        return None
    parsed = urlparse(base)
    path = parsed.path or "/0"
    if path in ("/", "/0"):
        path = "/1"
    return urlunparse(parsed._replace(path=path))


def create_chat_rate_limiter(
    config_dir: str = "config",
) -> RateLimiter:
    """优先 Redis（CHAT_RATE_LIMIT_REDIS_URL / REDIS_URL），否则进程内限流。"""
    cfg = load_chat_rate_limit_config(config_dir)
    if not cfg.enabled:
        return SlidingWindowRateLimiter(cfg)

    redis_url = _resolve_rate_limit_redis_url()
    if redis_url:
        try:
            import redis

            client = redis.from_url(redis_url, decode_responses=False)
            client.ping()
            return RedisRateLimiter(cfg, client)
        except Exception:
            pass
    return SlidingWindowRateLimiter(cfg)
