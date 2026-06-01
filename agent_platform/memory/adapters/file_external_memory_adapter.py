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
