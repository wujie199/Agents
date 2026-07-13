# -*- coding: utf-8 -*-
"""L1 热记忆：LangGraph BaseStore namespace（生产对齐 MEMORY_DESIGN §3.3）。"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from core.domain.context import RequestContext
from core.ports.memory import MemoryDelta, PromptMemorySnapshot

from agent_platform.memory.adapters.hot_memory_file_adapter import HotMemoryFileAdapter

logger = logging.getLogger(__name__)

_MEMORY_NS = "memory"
_PENDING_NS = "memory_pending"


def _memory_namespace(tenant_id: str) -> tuple[str, ...]:
    return (_MEMORY_NS, tenant_id)


def _pending_namespace(tenant_id: str) -> tuple[str, ...]:
    return (_PENDING_NS, tenant_id)


class HotMemoryLangGraphStoreAdapter(HotMemoryFileAdapter):
    """
    LangGraph Store 版 L1：namespace (memory, tenant_id) + key global|user_id。
    pending 存 (memory_pending, tenant_id) + key user_id。
    """

    def __init__(
        self,
        store: Any,
        *,
        hot_memory_max_chars: int = 2200,
        user_memory_max_chars: int = 1375,
    ):
        # 不创建文件目录；仅复用 HotMemoryFileAdapter 的 KV 逻辑
        self._store = store
        self._hot_memory_max = hot_memory_max_chars
        self._user_memory_max = user_memory_max_chars
        self._use_file_lock = False
        self._logger = logging.getLogger(__name__)
        self._memory_cache: dict[str, str] = {}
        self._user_cache: dict[str, str] = {}
        self._snapshot_hash: dict[str, str] = {}

    def _store_get(self, namespace: tuple[str, ...], key: str) -> str:
        try:
            item = self._store.get(namespace, key)
        except Exception as exc:
            self._logger.debug("store.get failed ns=%s key=%s: %s", namespace, key, exc)
            return ""
        if item is None:
            return ""
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            return str(value.get("content") or "")
        return str(value or "")

    def _store_put(self, namespace: tuple[str, ...], key: str, content: str) -> None:
        try:
            self._store.put(namespace, key, {"content": content})
        except Exception as exc:
            self._logger.warning(
                "store.put failed ns=%s key=%s: %s", namespace, key, exc
            )

    def _load_memory(self, tenant_id: str) -> str:
        if tenant_id in self._memory_cache:
            return self._memory_cache[tenant_id]
        content = self._store_get(_memory_namespace(tenant_id), "global")
        self._memory_cache[tenant_id] = content
        return content

    def _load_user(self, tenant_id: str, user_id: str) -> str:
        cache_key = f"{tenant_id}:{user_id}"
        if cache_key in self._user_cache:
            return self._user_cache[cache_key]
        content = self._store_get(_memory_namespace(tenant_id), user_id)
        self._user_cache[cache_key] = content
        return content

    def _save_memory(self, tenant_id: str, content: str) -> None:
        self._store_put(_memory_namespace(tenant_id), "global", content)
        self._memory_cache[tenant_id] = content

    def _save_user(self, tenant_id: str, user_id: str, content: str) -> None:
        self._store_put(_memory_namespace(tenant_id), user_id, content)
        self._user_cache[f"{tenant_id}:{user_id}"] = content

    def get_raw_memory(self, tenant_id: str) -> str:
        return self._load_memory(tenant_id)

    def get_raw_user(self, tenant_id: str, user_id: str) -> str:
        return self._load_user(tenant_id, user_id)

    def save_memory(self, tenant_id: str, content: str) -> None:
        self._save_memory(tenant_id, content)

    def save_user(self, tenant_id: str, user_id: str, content: str) -> None:
        self._save_user(tenant_id, user_id, content)

    def list_pending_deltas(self, tenant_id: str, user_id: str) -> List[MemoryDelta]:
        raw = self._store_get(_pending_namespace(tenant_id), user_id)
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("deltas") or []
        else:
            return []
        out: List[MemoryDelta] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()
            value = str(item.get("value") or "").strip()
            if key and value:
                out.append(
                    MemoryDelta(
                        key=key,
                        value=value,
                        source=str(item.get("source") or "user"),
                    )
                )
        return out

    def queue_pending_delta(
        self, tenant_id: str, user_id: str, delta: MemoryDelta
    ) -> None:
        merged = {d.key: d for d in self.list_pending_deltas(tenant_id, user_id)}
        merged[delta.key] = delta
        payload = [
            {"key": d.key, "value": d.value, "source": d.source}
            for d in merged.values()
        ]
        self._store_put(
            _pending_namespace(tenant_id),
            user_id,
            json.dumps({"deltas": payload}, ensure_ascii=False),
        )

    def flush_pending_deltas(
        self, tenant_id: str, user_id: str
    ) -> List[MemoryDelta]:
        deltas = self.list_pending_deltas(tenant_id, user_id)
        self._store_put(_pending_namespace(tenant_id), user_id, "")
        return deltas

    def purge_user_pending(self, tenant_id: str, user_id: str) -> None:
        self._store_put(_pending_namespace(tenant_id), user_id, "")

    def list_user_ids(self, tenant_id: str) -> List[str]:
        list_ns = getattr(self._store, "list_namespaces", None)
        if not callable(list_ns):
            return []
        try:
            rows = list_ns(
                prefix=(_MEMORY_NS, tenant_id),
                max_depth=2,
                limit=500,
            )
        except Exception:
            return []
        ids: List[str] = []
        for row in rows:
            ns = getattr(row, "namespace", None) or row
            if isinstance(ns, (list, tuple)) and len(ns) >= 2:
                key = str(ns[-1]) if len(ns) > 2 else ""
                if key and key != "global":
                    ids.append(key)
        return sorted(set(ids))

    def invalidate_cache(
        self, tenant_id: str, user_id: Optional[str] = None
    ) -> None:
        self._memory_cache.pop(tenant_id, None)
        if user_id:
            self._user_cache.pop(f"{tenant_id}:{user_id}", None)
            self._snapshot_hash.pop(f"{tenant_id}:{user_id}", None)
        else:
            prefix = f"{tenant_id}:"
            for k in list(self._user_cache):
                if k.startswith(prefix):
                    self._user_cache.pop(k, None)
            for k in list(self._snapshot_hash):
                if k.startswith(prefix):
                    self._snapshot_hash.pop(k, None)


def build_langgraph_memory_store(
    cfg: dict[str, Any],
    *,
    database_url: Optional[str] = None,
) -> Any:
    """按配置创建 LangGraph Store（Postgres 优先，否则 InMemory）。"""
    from agent_platform.memory.adapters.l1_langgraph_store_registry import (
        get_postgres_memory_store,
        setup_postgres_memory_store,
    )

    url = database_url or cfg.get("langgraph_store_database_url")
    if url:
        existing = get_postgres_memory_store()
        if existing is not None:
            return existing
        try:
            return setup_postgres_memory_store(str(url))
        except Exception as exc:
            logger.warning(
                "PostgresStore unavailable (%s), fallback InMemoryStore", exc
            )
    from langgraph.store.memory import InMemoryStore

    return InMemoryStore()
