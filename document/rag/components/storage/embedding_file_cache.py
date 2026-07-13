"""离线 embedding 向量文件缓存（JSON，按 key 分文件）。"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger("rag.embedding.file_cache")


def _safe_filename(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return digest[:32]


class EmbeddingFileCacheAdapter:
    """CachePort 最小实现，供离线建库 embedding 读/写缓存。"""

    def __init__(self, cache_dir: str | Path):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{_safe_filename(key)}.json"

    def get(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            _log.warning("读取 embedding 缓存失败 key=%s: %s", key[:24], exc)
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            _log.warning("写入 embedding 缓存失败 key=%s: %s", key[:24], exc)

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.is_file():
            path.unlink(missing_ok=True)

    def expire(self, key: str, ttl_seconds: int) -> None:
        return None

    def invalidate_pattern(self, pattern: str) -> int:
        return 0

    def build_key(self, tenant_id: str, category: str, identifier: str) -> str:
        return f"{tenant_id}:{category}:{identifier}"
