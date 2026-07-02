"""File/HTTP L4 外部画像 Provider。"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from core.domain.context import RequestContext
from core.ports.memory.provider import BaseMemoryProvider, MemorySnippet

logger = logging.getLogger(__name__)

_KV_RE = re.compile(r"^([^:]+):\s*(.+)$")


class FileProfileProvider(BaseMemoryProvider):
    """将 ExternalMemoryProvider 适配为 MemoryProvider。"""

    def __init__(self, external_memory: Any, *, max_prefetch_chars: int = 2000):
        self._external = external_memory
        self._max_prefetch_chars = max_prefetch_chars
        self._initialized = False

    @property
    def name(self) -> str:
        return "file_profile"

    def is_available(self) -> bool:
        return self._external is not None

    async def initialize(self, context: RequestContext) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    async def prefetch_turn(
        self, user_message: str, context: RequestContext
    ) -> list[MemorySnippet]:
        if not self.is_available():
            return []
        try:
            facts = await self._external.fetch_profile_facts(
                context.user_id, context.tenant_id
            )
        except Exception as exc:
            logger.warning("L4 prefetch fetch_profile_facts failed: %s", exc)
            raise
        if not facts:
            return []
        lines: list[str] = []
        total = 0
        for fact in facts:
            line = f"{fact.key}: {fact.value}"
            if total + len(line) + 1 > self._max_prefetch_chars:
                break
            lines.append(line)
            total += len(line) + 1
        if not lines:
            return []
        return [
            MemorySnippet(
                content="\n".join(lines),
                source=self.name,
                metadata={"fact_count": len(lines)},
            )
        ]

    async def search(
        self, query: str, context: RequestContext, *, limit: int = 5
    ) -> list[MemorySnippet]:
        snippets = await self.prefetch_turn(query, context)
        if not query.strip():
            return snippets[:limit]
        q = query.lower()
        filtered: list[MemorySnippet] = []
        for snip in snippets:
            if q in snip.content.lower():
                filtered.append(snip)
        return filtered[:limit] if filtered else snippets[:limit]

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "fetch_profile_facts",
                    "description": "读取当前用户 L4 外部画像 facts。",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "resolve_entity",
                    "description": "解析实体别名（CRM/LDAP 画像）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mention": {"type": "string", "description": "实体提及"}
                        },
                        "required": ["mention"],
                    },
                },
            },
        ]

    async def on_memory_write(
        self,
        tenant_id: str,
        user_id: str,
        action: str,
        payload: dict[str, Any],
    ) -> None:
        if not self.is_available():
            return
        target = str(payload.get("target") or "memory")
        if target != "user":
            return
        facts = self._payload_to_facts(action, payload)
        if not facts:
            return
        try:
            await self._external.upsert_profile_facts(tenant_id, user_id, facts)
        except Exception as exc:
            logger.warning("L4 on_memory_write mirror failed: %s", exc)

    @staticmethod
    def _payload_to_facts(action: str, payload: dict[str, Any]) -> list[dict[str, str]]:
        if action == "batch":
            facts: list[dict[str, str]] = []
            for op in payload.get("operations") or []:
                if not isinstance(op, dict):
                    continue
                facts.extend(
                    FileProfileProvider._payload_to_facts(
                        str(op.get("action") or ""), op
                    )
                )
            return facts

        content = str(payload.get("content") or "").strip()
        old_text = str(payload.get("old_text") or "").strip()

        if action == "add" and content:
            parsed = _KV_RE.match(content)
            if parsed:
                return [
                    {
                        "key": parsed.group(1).strip(),
                        "value": parsed.group(2).strip(),
                        "source": "l1_mirror",
                    }
                ]
            return [{"key": content[:80], "value": content, "source": "l1_mirror"}]

        if action == "replace" and content:
            parsed = _KV_RE.match(content)
            if parsed:
                return [
                    {
                        "key": parsed.group(1).strip(),
                        "value": parsed.group(2).strip(),
                        "source": "l1_mirror",
                    }
                ]
            if old_text:
                return [{"key": old_text[:80], "value": content, "source": "l1_mirror"}]

        if action == "remove" and old_text:
            parsed = _KV_RE.match(old_text)
            key = parsed.group(1).strip() if parsed else old_text[:80]
            return [{"key": key, "value": "", "source": "l1_mirror", "delete": "true"}]

        return []
