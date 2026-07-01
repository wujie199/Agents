"""离线建库索引清单：按文件 MD5 记录已入库文档，支持已索引跳过。"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_log = logging.getLogger("document.rag.index_manifest")

MANIFEST_VERSION = 1


def file_md5_hex(file_path: Path) -> str:
    """计算文件 MD5（二进制流）。"""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def doc_id_from_file_md5(file_md5: str) -> str:
    """由文件 MD5 生成稳定 doc_id，便于重复建库时 upsert 覆盖。"""
    return f"doc_{file_md5[:16]}"


class IndexManifest:
    """
    持久化清单：{data_dir}/indexed_by_md5.json
    结构: tenants -> tenant_id -> file_md5 -> entry
    """

    def __init__(self, manifest_path: Path):
        self._path = manifest_path
        self._data: Dict[str, Any] = self._load()

    @classmethod
    def for_data_dir(cls, data_dir: Path) -> "IndexManifest":
        return cls(data_dir / "indexed_by_md5.json")

    def _load(self) -> Dict[str, Any]:
        if not self._path.is_file():
            return {"version": MANIFEST_VERSION, "tenants": {}}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f) or {}
            if "tenants" not in raw:
                raw["tenants"] = {}
            raw.setdefault("version", MANIFEST_VERSION)
            return raw
        except (json.JSONDecodeError, OSError) as exc:
            _log.warning("读取索引清单失败，将重建: %s", exc)
            return {"version": MANIFEST_VERSION, "tenants": {}}

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._path)

    def get_entry(self, tenant_id: str, file_md5: str) -> Optional[Dict[str, Any]]:
        return self._data.get("tenants", {}).get(tenant_id, {}).get(file_md5)

    def is_indexed(self, tenant_id: str, file_md5: str) -> bool:
        return self.get_entry(tenant_id, file_md5) is not None

    def has_tenant(self, tenant_id: str) -> bool:
        """租户是否已有至少一条索引记录。"""
        return bool(self._data.get("tenants", {}).get(tenant_id))

    def matches_index_config(
        self,
        tenant_id: str,
        file_md5: str,
        *,
        model_version: str,
        config_hash: str,
    ) -> bool:
        """文件 MD5 已索引且 model_version / config_hash 与当前配置一致。"""
        entry = self.get_entry(tenant_id, file_md5)
        if entry is None:
            return False
        if entry.get("model_version") != model_version:
            return False
        stored_hash = entry.get("config_hash")
        if stored_hash is None:
            return False
        return stored_hash == config_hash

    def register(
        self,
        tenant_id: str,
        file_md5: str,
        *,
        doc_id: str,
        source_path: str,
        model_version: str,
        config_hash: str,
        chunk_count: int = 0,
        vectors_written: int = 0,
    ) -> None:
        tenants = self._data.setdefault("tenants", {})
        bucket = tenants.setdefault(tenant_id, {})
        bucket[file_md5] = {
            "doc_id": doc_id,
            "source_path": source_path,
            "file_md5": file_md5,
            "model_version": model_version,
            "config_hash": config_hash,
            "chunk_count": chunk_count,
            "vectors_written": vectors_written,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save()

    def remove(self, tenant_id: str, file_md5: str) -> None:
        tenants = self._data.get("tenants", {})
        bucket = tenants.get(tenant_id, {})
        if file_md5 in bucket:
            del bucket[file_md5]
            self.save()
