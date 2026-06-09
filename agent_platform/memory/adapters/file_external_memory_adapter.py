import logging
from pathlib import Path
from typing import List, Optional

import yaml

from core.domain.context import RequestContext
from core.ports.external_memory import Entity, Fact


class FileExternalMemoryAdapter:
    """L4 外部画像：从 YAML 文件加载实体与结构化事实（开发/测试用）。"""

    def __init__(self, profiles_dir: str = "data/external_profiles"):
        self._profiles_dir = Path(profiles_dir)
        self._logger = logging.getLogger(__name__)

    def _profile_path(self, tenant_id: str, user_id: str) -> Path:
        return self._profiles_dir / tenant_id / f"{user_id}.yaml"

    def _load_profile(self, tenant_id: str, user_id: str) -> dict:
        path = self._profile_path(tenant_id, user_id)
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _save_profile_file(
        self, tenant_id: str, user_id: str, profile: dict
    ) -> None:
        path = self._profile_path(tenant_id, user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                profile,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )

    async def resolve_entity(
        self, mention: str, ctx: RequestContext
    ) -> Optional[Entity]:
        profile = self._load_profile(ctx.tenant_id, ctx.user_id)
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
        profile = self._load_profile(tenant_id, user_id)
        facts: List[Fact] = []
        for item in profile.get("facts") or []:
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

    async def list_profile_users(self, tenant_id: str) -> List[str]:
        tenant_dir = self._profiles_dir / tenant_id
        if not tenant_dir.is_dir():
            return []
        return sorted(p.stem for p in tenant_dir.glob("*.yaml"))

    async def get_profile(self, tenant_id: str, user_id: str) -> dict:
        return self._load_profile(tenant_id, user_id)

    async def save_profile(
        self, tenant_id: str, user_id: str, profile: dict
    ) -> None:
        self._save_profile_file(tenant_id, user_id, profile or {})

    async def upsert_profile_facts(
        self, tenant_id: str, user_id: str, facts: List[dict]
    ) -> int:
        profile = self._load_profile(tenant_id, user_id)
        by_key: dict[str, dict] = {}
        for item in profile.get("facts") or []:
            if isinstance(item, dict) and item.get("key"):
                by_key[str(item["key"])] = item
        updated = 0
        for item in facts:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            if not key:
                continue
            value = item.get("value")
            if value is None:
                continue
            by_key[str(key)] = {
                "key": str(key),
                "value": str(value),
                "source": str(item.get("source", "external")),
            }
            updated += 1
        profile["facts"] = list(by_key.values())
        self._save_profile_file(tenant_id, user_id, profile)
        return updated

    async def delete_profile(self, tenant_id: str, user_id: str) -> bool:
        path = self._profile_path(tenant_id, user_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    async def purge_tenant_profiles(self, tenant_id: str) -> int:
        tenant_dir = self._profiles_dir / tenant_id
        if not tenant_dir.is_dir():
            return 0
        deleted = 0
        for path in tenant_dir.glob("*.yaml"):
            path.unlink()
            deleted += 1
        if deleted and not any(tenant_dir.iterdir()):
            tenant_dir.rmdir()
        return deleted
