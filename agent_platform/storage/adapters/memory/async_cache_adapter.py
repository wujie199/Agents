import fnmatch
import time
from typing import Any, Dict, Optional, Tuple


class AsyncMemoryCacheAdapter:
    """开发/测试用异步内存缓存，兼容 RAG 的 await get/set/invalidate_pattern。"""

    def __init__(self, prefix: str = "agents"):
        self._prefix = prefix
        self._cache: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}

    def _full_key(self, key: str) -> str:
        if key.startswith(f"{self._prefix}:"):
            return key
        return f"{self._prefix}:{key}"

    def _purge_expired(self, key: str) -> None:
        exp = self._expiry.get(key)
        if exp is not None and time.time() > exp:
            self._cache.pop(key, None)
            self._expiry.pop(key, None)

    async def get(self, key: str) -> Optional[Any]:
        full = self._full_key(key)
        self._purge_expired(full)
        return self._cache.get(full)

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        ttl: Optional[int] = None,
    ) -> None:
        ttl_seconds = ttl_seconds if ttl_seconds is not None else ttl
        full = self._full_key(key)
        self._cache[full] = value
        if ttl_seconds:
            self._expiry[full] = time.time() + ttl_seconds
        else:
            self._expiry.pop(full, None)

    async def delete(self, key: str) -> None:
        full = self._full_key(key)
        self._cache.pop(full, None)
        self._expiry.pop(full, None)

    async def expire(self, key: str, ttl_seconds: int) -> None:
        full = self._full_key(key)
        if full in self._cache:
            self._expiry[full] = time.time() + ttl_seconds

    async def invalidate_pattern(self, pattern: str) -> int:
        full_pattern = self._full_key(pattern)
        keys = [k for k in list(self._cache.keys()) if fnmatch.fnmatch(k, full_pattern)]
        for key in keys:
            self._cache.pop(key, None)
            self._expiry.pop(key, None)
        return len(keys)

    def build_key(self, tenant_id: str, category: str, identifier: str) -> str:
        return f"{tenant_id}:{self._prefix}:{category}:{identifier}"

    def health(self) -> dict:
        return {"status": "healthy", "type": "async_memory_cache", "keys": len(self._cache)}
