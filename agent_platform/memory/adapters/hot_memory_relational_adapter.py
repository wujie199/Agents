# -*- coding: utf-8 -*-
"""L1 热记忆：与 L2 同库存储（hot_memory_docs 表）。"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.domain.context import RequestContext
from core.ports.memory import MemoryDelta, PromptMemorySnapshot

from agent_platform.memory.adapters.hot_memory_file_adapter import HotMemoryFileAdapter


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
        has_loop = True
    except RuntimeError:
        has_loop = False
    if not has_loop:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class HotMemoryRelationalAdapter:
    """与 HotMemoryFileAdapter 同 API，持久化到 archive DB。"""

    def __init__(
        self,
        archive_db: Any,
        hot_memory_max_chars: int = 2200,
        user_memory_max_chars: int = 1375,
    ):
        self._db = archive_db
        self._hot_memory_max = hot_memory_max_chars
        self._user_memory_max = user_memory_max_chars
        self._logger = logging.getLogger(__name__)
        self._memory_cache: Dict[str, str] = {}
        self._user_cache: Dict[str, str] = {}
        self._snapshot_hash: Dict[str, str] = {}

    truncate = staticmethod(HotMemoryFileAdapter.truncate)
    _compute_hash = staticmethod(HotMemoryFileAdapter._compute_hash)
    _parse_kv_line = staticmethod(HotMemoryFileAdapter._parse_kv_line)
    _apply_kv_updates = staticmethod(HotMemoryFileAdapter._apply_kv_updates)

    def _apply_user_kv_updates(self, content, updates, sources=None):
        return self._apply_kv_updates(content, updates, sources)

    def _load_memory(self, tenant_id: str) -> str:
        if tenant_id in self._memory_cache:
            return self._memory_cache[tenant_id]
        content = (
            _run_async(self._db.get_hot_memory_doc(tenant_id, "memory")) or ""
        )
        self._memory_cache[tenant_id] = content
        return content

    def _load_user(self, tenant_id: str, user_id: str) -> str:
        key = f"{tenant_id}:{user_id}"
        if key in self._user_cache:
            return self._user_cache[key]
        content = (
            _run_async(
                self._db.get_hot_memory_doc(tenant_id, "user", user_id)
            )
            or ""
        )
        self._user_cache[key] = content
        return content

    def _save_memory(self, tenant_id: str, content: str) -> None:
        _run_async(
            self._db.upsert_hot_memory_doc(tenant_id, "memory", content)
        )
        self._memory_cache[tenant_id] = content

    def _save_user(self, tenant_id: str, user_id: str, content: str) -> None:
        _run_async(
            self._db.upsert_hot_memory_doc(
                tenant_id, "user", content, user_id=user_id
            )
        )
        self._user_cache[f"{tenant_id}:{user_id}"] = content

    def compose_snapshot(self, context: RequestContext) -> PromptMemorySnapshot:
        memory_content = self.truncate(
            self._load_memory(context.tenant_id), self._hot_memory_max
        )
        user_content = self.truncate(
            self._load_user(context.tenant_id, context.user_id),
            self._user_memory_max,
        )
        combined = (
            f"# SYSTEM MEMORY\n\n{memory_content}\n\n"
            f"# USER PREFERENCES\n\n{user_content}"
        )
        snapshot_hash = self._compute_hash(combined)
        self._snapshot_hash[f"{context.tenant_id}:{context.user_id}"] = (
            snapshot_hash
        )
        return PromptMemorySnapshot(
            memory_text=combined, hash=snapshot_hash, frozen=True
        )

    def list_user_ids(self, tenant_id: str) -> List[str]:
        return _run_async(self._db.list_hot_memory_user_ids(tenant_id))

    def strip_user_keys(self, tenant_id: str, user_id: str, keys: List[str]) -> int:
        remove = {k for k in keys if k}
        if not remove:
            return 0
        current = self._load_user(tenant_id, user_id)
        if not current:
            return 0
        new_lines: List[str] = []
        removed = 0
        seen: set[str] = set()
        for line in current.splitlines():
            parsed = self._parse_kv_line(line)
            if parsed is None:
                new_lines.append(line)
                continue
            key, _ = parsed
            if key in remove:
                if key not in seen:
                    removed += 1
                seen.add(key)
                continue
            if key in seen:
                continue
            seen.add(key)
            new_lines.append(line)
        if removed:
            self._save_user(
                tenant_id,
                user_id,
                self.truncate("\n".join(new_lines), self._user_memory_max * 2),
            )
        return removed

    def merge_user_facts_upsert(
        self, tenant_id: str, user_id: str, deltas: List[MemoryDelta]
    ) -> List[dict]:
        if not deltas:
            return []
        current = self._load_user(tenant_id, user_id)
        updates = {d.key: d.value for d in deltas}
        sources = {d.key: d.source for d in deltas}
        new_content, changes = self._apply_user_kv_updates(
            current, updates, sources
        )
        if not changes:
            return changes
        self._save_user(
            tenant_id,
            user_id,
            self.truncate(new_content, self._user_memory_max * 2),
        )
        return changes

    def apply_delta(self, tenant_id: str, user_id: str, delta: MemoryDelta) -> None:
        if delta.source == "memory":
            current = self._load_memory(tenant_id)
            new_content, _ = self._apply_kv_updates(
                current, {delta.key: delta.value}, {delta.key: delta.source}
            )
            self._save_memory(
                tenant_id,
                self.truncate(new_content, self._hot_memory_max * 2),
            )
        elif delta.source in ("user", "external"):
            current = self._load_user(tenant_id, user_id)
            new_content, _ = self._apply_user_kv_updates(
                current, {delta.key: delta.value}, {delta.key: delta.source}
            )
            self._save_user(
                tenant_id,
                user_id,
                self.truncate(new_content, self._user_memory_max * 2),
            )
        else:
            self._logger.warning("Unknown memory source: %s", delta.source)

    def queue_pending_delta(
        self, tenant_id: str, user_id: str, delta: MemoryDelta
    ) -> None:
        pending = self.list_pending_deltas(tenant_id, user_id)
        pending.append(delta)
        lines = [
            json.dumps(
                {"key": d.key, "value": d.value, "source": d.source},
                ensure_ascii=False,
            )
            for d in pending
        ]
        _run_async(
            self._db.upsert_hot_memory_doc(
                tenant_id,
                "pending",
                "\n".join(lines) + ("\n" if lines else ""),
                user_id=user_id,
            )
        )

    def list_pending_deltas(self, tenant_id: str, user_id: str) -> List[MemoryDelta]:
        raw = (
            _run_async(
                self._db.get_hot_memory_doc(tenant_id, "pending", user_id)
            )
            or ""
        )
        deltas: List[MemoryDelta] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            deltas.append(
                MemoryDelta(
                    key=data["key"],
                    value=data["value"],
                    source=data.get("source", "user"),
                )
            )
        return deltas

    def flush_pending_deltas(self, tenant_id: str, user_id: str) -> List[MemoryDelta]:
        deltas = self.list_pending_deltas(tenant_id, user_id)
        _run_async(
            self._db.delete_hot_memory_doc(tenant_id, "pending", user_id)
        )
        return deltas

    def get_raw_memory(self, tenant_id: str) -> str:
        return self._load_memory(tenant_id)

    def get_raw_user(self, tenant_id: str, user_id: str) -> str:
        return self._load_user(tenant_id, user_id)

    def save_memory(self, tenant_id: str, content: str) -> None:
        self._save_memory(tenant_id, content)

    def save_user(self, tenant_id: str, user_id: str, content: str) -> None:
        self._save_user(tenant_id, user_id, content)

    def clear_user(self, tenant_id: str, user_id: str) -> None:
        _run_async(self._db.delete_hot_memory_doc(tenant_id, "user", user_id))
        _run_async(self._db.delete_hot_memory_doc(tenant_id, "pending", user_id))
        self.invalidate_cache(tenant_id, user_id)

    def invalidate_cache(
        self, tenant_id: str, user_id: Optional[str] = None
    ) -> None:
        self._memory_cache.pop(tenant_id, None)
        if user_id:
            self._user_cache.pop(f"{tenant_id}:{user_id}", None)
        else:
            prefix = f"{tenant_id}:"
            for key in list(self._user_cache):
                if key.startswith(prefix):
                    del self._user_cache[key]

    def get_snapshot_hash(self, tenant_id: str, user_id: str) -> Optional[str]:
        return self._snapshot_hash.get(f"{tenant_id}:{user_id}")

    @property
    def store_dir(self) -> Path:
        return Path("relational://hot_memory_docs")
