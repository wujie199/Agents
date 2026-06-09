"""L4 HTTP 外部画像适配器（REST 骨架，对接 CRM/LDAP 网关）。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from core.domain.context import RequestContext
from core.ports.external_memory import Entity, Fact


class HttpExternalMemoryAdapter:
    """REST API：`{base_url}/tenants/{tenant}/users/{user}/profile` 等。"""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        api_key: Optional[str] = None,
    ):
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._api_key = api_key
        self._logger = logging.getLogger(__name__)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[dict] = None,
    ) -> Any:
        url = f"{self._base}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = Request(url, data=data, headers=self._headers(), method=method)

        def _do() -> Any:
            try:
                with urlopen(req, timeout=self._timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    if method == "DELETE":
                        return True
                    if not raw:
                        return {}
                    return json.loads(raw)
            except HTTPError as e:
                if e.code == 404:
                    return None if method != "DELETE" else False
                raise

        try:
            return await asyncio.to_thread(_do)
        except URLError as e:
            self._logger.warning("HTTP external profile request failed: %s %s", url, e)
            raise

    def _tenant_path(self, tenant_id: str) -> str:
        return f"/tenants/{quote(tenant_id, safe='')}"

    def _user_path(self, tenant_id: str, user_id: str) -> str:
        return (
            f"{self._tenant_path(tenant_id)}/users/{quote(user_id, safe='')}"
        )

    @staticmethod
    def _facts_from_profile(profile: dict) -> List[Fact]:
        facts: List[Fact] = []
        for item in (profile or {}).get("facts") or []:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            value = item.get("value")
            if key and value is not None:
                facts.append(
                    Fact(
                        key=str(key),
                        value=str(value),
                        source=str(item.get("source", "external")),
                    )
                )
        return facts

    async def resolve_entity(
        self, mention: str, ctx: RequestContext
    ) -> Optional[Entity]:
        profile = await self.get_profile(ctx.tenant_id, ctx.user_id)
        if not profile:
            return None
        entities = profile.get("entities") or {}
        mention_lower = mention.lower()
        if mention in entities:
            data = entities[mention]
            return Entity(
                mention=mention,
                canonical_id=str(data.get("canonical_id", mention)),
                display_name=str(data.get("display_name", mention)),
            )
        for alias, data in entities.items():
            display = str(data.get("display_name", alias))
            if mention_lower in alias.lower() or mention_lower in display.lower():
                return Entity(
                    mention=mention,
                    canonical_id=str(data.get("canonical_id", alias)),
                    display_name=display,
                )
        return None

    async def fetch_profile_facts(
        self, user_id: str, tenant_id: str
    ) -> List[Fact]:
        profile = await self.get_profile(tenant_id, user_id)
        return self._facts_from_profile(profile or {})

    async def list_profile_users(self, tenant_id: str) -> List[str]:
        data = await self._request("GET", f"{self._tenant_path(tenant_id)}/users")
        if not data:
            return []
        users = data.get("users") if isinstance(data, dict) else data
        if not isinstance(users, list):
            return []
        return sorted(str(u) for u in users)

    async def get_profile(self, tenant_id: str, user_id: str) -> dict:
        data = await self._request(
            "GET", f"{self._user_path(tenant_id, user_id)}/profile"
        )
        return data or {}

    async def save_profile(
        self, tenant_id: str, user_id: str, profile: dict
    ) -> None:
        await self._request(
            "PUT",
            f"{self._user_path(tenant_id, user_id)}/profile",
            body=profile or {},
        )

    async def upsert_profile_facts(
        self, tenant_id: str, user_id: str, facts: List[dict]
    ) -> int:
        profile = await self.get_profile(tenant_id, user_id)
        by_key: dict[str, dict] = {}
        for item in profile.get("facts") or []:
            if isinstance(item, dict) and item.get("key"):
                by_key[str(item["key"])] = item
        updated = 0
        for item in facts:
            if not isinstance(item, dict) or not item.get("key"):
                continue
            value = item.get("value")
            if value is None:
                continue
            key = str(item["key"])
            by_key[key] = {
                "key": key,
                "value": str(value),
                "source": str(item.get("source", "external")),
            }
            updated += 1
        profile = profile or {}
        profile["facts"] = list(by_key.values())
        await self.save_profile(tenant_id, user_id, profile)
        return updated

    async def delete_profile(self, tenant_id: str, user_id: str) -> bool:
        result = await self._request(
            "DELETE", f"{self._user_path(tenant_id, user_id)}/profile"
        )
        return bool(result)

    async def purge_tenant_profiles(self, tenant_id: str) -> int:
        users = await self.list_profile_users(tenant_id)
        deleted = 0
        for user_id in users:
            if await self.delete_profile(tenant_id, user_id):
                deleted += 1
        return deleted
