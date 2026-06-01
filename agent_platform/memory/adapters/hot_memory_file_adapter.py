import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from core.domain.context import RequestContext
from core.ports.memory import MemoryDelta, PromptMemorySnapshot


class HotMemoryFileAdapter:
    """L1 热记忆：Markdown 文件 + 进程内缓存。"""

    def __init__(
        self,
        store_dir: str = "workspace/memory",
        hot_memory_max_chars: int = 2200,
        user_memory_max_chars: int = 1375,
    ):
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._hot_memory_max = hot_memory_max_chars
        self._user_memory_max = user_memory_max_chars
        self._logger = logging.getLogger(__name__)

        self._memory_cache: Dict[str, str] = {}
        self._user_cache: Dict[str, str] = {}
        self._snapshot_hash: Dict[str, str] = {}

    def _memory_path(self, tenant_id: str) -> Path:
        path = self._store_dir / tenant_id / "MEMORY.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _user_path(self, tenant_id: str, user_id: str) -> Path:
        path = self._store_dir / tenant_id / f"USER_{user_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _pending_path(self, tenant_id: str, user_id: str) -> Path:
        path = self._store_dir / tenant_id / f"pending_{user_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _load_memory(self, tenant_id: str) -> str:
        if tenant_id in self._memory_cache:
            return self._memory_cache[tenant_id]
        path = self._memory_path(tenant_id)
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        self._memory_cache[tenant_id] = content
        return content

    def _load_user(self, tenant_id: str, user_id: str) -> str:
        cache_key = f"{tenant_id}:{user_id}"
        if cache_key in self._user_cache:
            return self._user_cache[cache_key]
        path = self._user_path(tenant_id, user_id)
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        self._user_cache[cache_key] = content
        return content

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)

    def _save_memory(self, tenant_id: str, content: str) -> None:
        path = self._memory_path(tenant_id)
        self._atomic_write(path, content)
        self._memory_cache[tenant_id] = content

    def _save_user(self, tenant_id: str, user_id: str, content: str) -> None:
        path = self._user_path(tenant_id, user_id)
        self._atomic_write(path, content)
        self._user_cache[f"{tenant_id}:{user_id}"] = content

    @staticmethod
    def truncate(content: str, max_chars: int) -> str:
        if len(content) <= max_chars:
            return content
        truncated = content[:max_chars]
        last_newline = truncated.rfind("\n")
        if last_newline > max_chars * 0.8:
            truncated = truncated[:last_newline]
        return truncated + "\n[... truncated ...]"

    @staticmethod
    def _compute_hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

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
        cache_key = f"{context.tenant_id}:{context.user_id}"
        self._snapshot_hash[cache_key] = snapshot_hash
        return PromptMemorySnapshot(
            memory_text=combined,
            hash=snapshot_hash,
            frozen=True,
        )

    def apply_delta(
        self,
        tenant_id: str,
        user_id: str,
        delta: MemoryDelta,
    ) -> None:
        line = f"{delta.key}: {delta.value}"
        if delta.source == "memory":
            current = self._load_memory(tenant_id)
            updated = self.truncate(
                f"{current}\n\n{line}".strip(), self._hot_memory_max * 2
            )
            self._save_memory(tenant_id, updated)
        elif delta.source == "user":
            current = self._load_user(tenant_id, user_id)
            updated = self.truncate(
                f"{current}\n\n{line}".strip(), self._user_memory_max * 2
            )
            self._save_user(tenant_id, user_id, updated)
        else:
            self._logger.warning("Unknown memory source: %s", delta.source)

    def queue_pending_delta(
        self, tenant_id: str, user_id: str, delta: MemoryDelta
    ) -> None:
        path = self._pending_path(tenant_id, user_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"key": delta.key, "value": delta.value, "source": delta.source},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def list_pending_deltas(
        self, tenant_id: str, user_id: str
    ) -> List[MemoryDelta]:
        path = self._pending_path(tenant_id, user_id)
        if not path.exists():
            return []
        deltas: List[MemoryDelta] = []
        for line in path.read_text(encoding="utf-8").splitlines():
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

    def flush_pending_deltas(
        self, tenant_id: str, user_id: str
    ) -> List[MemoryDelta]:
        deltas = self.list_pending_deltas(tenant_id, user_id)
        path = self._pending_path(tenant_id, user_id)
        if path.exists():
            path.unlink()
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
        path = self._user_path(tenant_id, user_id)
        if path.exists():
            path.unlink()
        pending = self._pending_path(tenant_id, user_id)
        if pending.exists():
            pending.unlink()
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
        return self._store_dir
