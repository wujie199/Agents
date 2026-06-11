"""L4 外部画像 TTL 缓存装饰器（Redis 或进程内）。"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, List, Optional, Tuple

from core.domain.context import RequestContext
from core.ports.external_memory import Entity, Fact


class CachedExternalMemoryAdapter:
    """包装 ExternalMemoryProvider；读路径 TTL 缓存（多 Pod 用 Redis）。"""

    def __init__(
        self,
        inner: Any,
        ttl_seconds: int = 300,
        *,
        cache_port: Any = None,
    ):
        self._inner = inner
        self._ttl = max(1, int(ttl_seconds))
        self._cache_port = cache_port
        self._local: dict[str, Tuple[float, Any]] = {}
        self._stats = {"hit": 0, "miss": 0}

    @property
    def cache_stats(self) -> dict[str, int]:
        return dict(self._stats)

    def _entity_to_dict(self, entity: Optional[Entity]) -> Optional[dict]:
        if entity is None:
            return None
        return asdict(entity)

    def _entity_from_dict(self, data: Any) -> Optional[Entity]:
        if data is None:
            return None
        if isinstance(data, Entity):
            return data
        if isinstance(data, dict):
            return Entity(
                mention=str(data.get("mention", "")),
                canonical_id=str(data.get("canonical_id", "")),
                display_name=str(data.get("display_name", "")),
            )
        return None

    def _facts_to_dicts(self, facts: List[Fact]) -> List[dict]:
        return [asdict(f) if isinstance(f, Fact) else dict(f) for f in facts]

    def _facts_from_dicts(self, data: Any) -> List[Fact]:
        if not data:
            return []
        out: List[Fact] = []
        for item in data:
            if isinstance(item, Fact):
                out.append(item)
            elif isinstance(item, dict):
                out.append(
                    Fact(
                        key=str(item.get("key", "")),
                        value=str(item.get("value", "")),
                        source=str(item.get("source", "")),
                    )
                )
        return out

    async def _cache_get(self, key: str) -> Tuple[bool, Any]:
        if self._cache_port is not None:
            try:
                value = await self._cache_port.get(key)
                if value is not None:
                    self._stats["hit"] += 1
                    return True, value
                self._stats["miss"] += 1
                return False, None
            except Exception:
                pass
        entry = self._local.get(key)
        if entry is None:
            self._stats["miss"] += 1
            return False, None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            del self._local[key]
            self._stats["miss"] += 1
            return False, None
        self._stats["hit"] += 1
        return True, value

    async def _cache_set(self, key: str, value: Any) -> None:
        if self._cache_port is not None:
            try:
                await self._cache_port.set(key, value, ttl_seconds=self._ttl)
                return
            except Exception:
                pass
        self._local[key] = (time.monotonic() + self._ttl, value)

    async def _invalidate_pattern(self, pattern: str) -> None:
        if self._cache_port is not None and hasattr(
            self._cache_port, "invalidate_pattern"
        ):
            try:
                await self._cache_port.invalidate_pattern(pattern)
            except Exception:
                pass
        prefix = pattern.rstrip("*")
        for key in list(self._local):
            if key.startswith(prefix):
                del self._local[key]

    async def _invalidate_user(self, tenant_id: str, user_id: str) -> None:
        await self._invalidate_pattern(f"{tenant_id}:{user_id}:*")
        await self._invalidate_pattern(f"tenant_users:{tenant_id}")

    async def invalidate_user_profile_cache(
        self, tenant_id: str, user_id: str
    ) -> None:
        """会话中强制刷新 L4 画像前调用。"""
        await self._invalidate_user(tenant_id, user_id)

    async def _invalidate_tenant(self, tenant_id: str) -> None:
        await self._invalidate_pattern(f"{tenant_id}:*")
        await self._invalidate_pattern(f"tenant_users:{tenant_id}")

    def invalidate_all(self) -> None:
        self._local.clear()

    async def resolve_entity(
        self, mention: str, ctx: RequestContext
    ) -> Optional[Entity]:
        key = f"{ctx.tenant_id}:{ctx.user_id}:entity:{mention}"
        hit, value = await self._cache_get(key)
        if hit:
            return self._entity_from_dict(value)
        entity = await self._inner.resolve_entity(mention, ctx)
        await self._cache_set(key, self._entity_to_dict(entity))
        return entity

    async def fetch_profile_facts(
        self, user_id: str, tenant_id: str
    ) -> List[Fact]:
        key = f"{tenant_id}:{user_id}:facts"
        hit, value = await self._cache_get(key)
        if hit:
            return self._facts_from_dicts(value)
        facts = await self._inner.fetch_profile_facts(user_id, tenant_id)
        await self._cache_set(key, self._facts_to_dicts(facts))
        return facts

    async def list_profile_users(self, tenant_id: str) -> List[str]:
        key = f"tenant_users:{tenant_id}"
        hit, value = await self._cache_get(key)
        if hit:
            return list(value or [])
        users = await self._inner.list_profile_users(tenant_id)
        await self._cache_set(key, users)
        return users

    async def get_profile(self, tenant_id: str, user_id: str) -> dict:
        key = f"{tenant_id}:{user_id}:profile"
        hit, value = await self._cache_get(key)
        if hit:
            return dict(value or {})
        profile = await self._inner.get_profile(tenant_id, user_id)
        await self._cache_set(key, profile)
        return profile

    async def save_profile(
        self, tenant_id: str, user_id: str, profile: dict
    ) -> None:
        await self._inner.save_profile(tenant_id, user_id, profile)
        await self._invalidate_user(tenant_id, user_id)

    async def upsert_profile_facts(
        self, tenant_id: str, user_id: str, facts: List[dict]
    ) -> int:
        count = await self._inner.upsert_profile_facts(tenant_id, user_id, facts)
        await self._invalidate_user(tenant_id, user_id)
        return count

    async def delete_profile(self, tenant_id: str, user_id: str) -> bool:
        deleted = await self._inner.delete_profile(tenant_id, user_id)
        await self._invalidate_user(tenant_id, user_id)
        return deleted

    async def purge_tenant_profiles(self, tenant_id: str) -> int:
        count = await self._inner.purge_tenant_profiles(tenant_id)
        await self._invalidate_tenant(tenant_id)
        return count
