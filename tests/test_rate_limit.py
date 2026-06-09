"""API 限流单元测试。"""

from __future__ import annotations

from app.api.rate_limit import ChatRateLimitConfig, SlidingWindowRateLimiter


def test_rate_limiter_allows_under_limit():
    limiter = SlidingWindowRateLimiter(
        ChatRateLimitConfig(
            enabled=True,
            tenant_requests_per_minute=3,
            user_requests_per_minute=2,
        )
    )
    ok, _ = limiter.check("t1", "u1")
    assert ok
    ok, _ = limiter.check("t1", "u1")
    assert ok


def test_rate_limiter_blocks_user():
    limiter = SlidingWindowRateLimiter(
        ChatRateLimitConfig(
            enabled=True,
            tenant_requests_per_minute=10,
            user_requests_per_minute=2,
        )
    )
    assert limiter.check("t1", "u1")[0]
    assert limiter.check("t1", "u1")[0]
    ok, reason = limiter.check("t1", "u1")
    assert not ok
    assert reason


def test_rate_limiter_disabled():
    limiter = SlidingWindowRateLimiter(ChatRateLimitConfig(enabled=False))
    for _ in range(5):
        assert limiter.check("t", "u")[0]


def test_redis_rate_limiter_blocks():
    from unittest.mock import MagicMock

    from app.api.rate_limit import RedisRateLimiter

    client = MagicMock()
    client.incr.side_effect = [1, 1, 2, 2, 3, 3]
    client.expire.return_value = True
    limiter = RedisRateLimiter(
        ChatRateLimitConfig(
            enabled=True,
            tenant_requests_per_minute=10,
            user_requests_per_minute=2,
        ),
        client,
    )
    assert limiter.check("t1", "u1")[0]
    assert limiter.check("t1", "u1")[0]
    ok, reason = limiter.check("t1", "u1")
    assert not ok
    assert reason
