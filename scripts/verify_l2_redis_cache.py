#!/usr/bin/env python3
"""验证 L2 session_search Redis 缓存：第 1 次写 Redis，第 2 次毫秒级命中。"""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


async def main() -> int:
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

    from core.domain.context import RequestContext
    from app.agents.context_factory import build_chat_run_context
    from core.ports.memory import TurnRecord

    session = f"redis_verify_{uuid.uuid4().hex[:8]}"
    req = RequestContext(
        tenant_id="tenant1",
        user_id="user1",
        session_id=session,
        trace_id="verify",
        channel="script",
    )
    ctx = build_chat_run_context(req, profile="dev", data_dir="data")
    memory = ctx.require_memory()
    cache = (ctx.extra or {}).get("cache")
    adapter = type(cache).__name__ if cache else "none"
    print(f"cache_adapter={adapter}")
    if adapter != "EnterpriseRedisCacheAdapter":
        print("FAIL: 未使用 Redis 缓存适配器")
        return 1

    await memory.persist_turn(
        req,
        TurnRecord(
            role="user",
            content="项目代号是 Phoenix",
            ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
        ),
    )

    query = "Phoenix"
    limit = 5
    scope = "session"
    raw = (
        f"{req.tenant_id}:{req.user_id}:{req.session_id}:{scope}:{query}:{limit}"
    )
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    cache_key = cache.build_key(req.tenant_id, "sess", digest)

    t0 = time.perf_counter()
    r1 = await memory.session_search(query, req, limit=limit, scope=scope)
    t1 = time.perf_counter()
    exists = await cache.exists(cache_key)
    t2 = time.perf_counter()
    r2 = await memory.session_search(query, req, limit=limit, scope=scope)
    t3 = time.perf_counter()

    ms1 = round((t1 - t0) * 1000)
    ms2 = round((t3 - t2) * 1000)
    print(f"cache_key={cache_key}")
    print(f"after 1st: redis_exists={exists}")
    print(f"1st_ms={ms1}  2nd_ms={ms2}  identical={r1 == r2}")

    if exists and r1 == r2 and ms2 < max(ms1 // 2, 50):
        print("PASS: Redis 缓存生效（第 1 次写入，第 2 次快速命中）")
        return 0

    print("FAIL: 未满足 Redis 缓存预期")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
