"""L3 Skill 运行时元数据侧写（published 只读；按 tenant 隔离）。"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class SkillMetaStore:
    def __init__(self, meta_dir: str = "skills/meta"):
        self._dir = Path(meta_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)

    def _tenant_dir(self, tenant_id: str) -> Path:
        tid = tenant_id or "default"
        path = self._dir / tid
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _path(self, skill_id: str, tenant_id: str = "default") -> Path:
        return self._tenant_dir(tenant_id) / f"{skill_id}.json"

    def _legacy_path(self, skill_id: str) -> Path:
        return self._dir / f"{skill_id}.json"

    def _migrate_legacy(self, skill_id: str, tenant_id: str) -> None:
        legacy = self._legacy_path(skill_id)
        if not legacy.exists():
            return
        target = self._path(skill_id, tenant_id)
        if target.exists():
            return
        try:
            shutil.copy2(legacy, target)
            self._logger.info("Migrated skill meta %s -> %s", legacy, target)
        except Exception as e:
            self._logger.warning("Migrate skill meta failed %s: %s", skill_id, e)

    def load(self, skill_id: str, tenant_id: str = "default") -> dict:
        self._migrate_legacy(skill_id, tenant_id)
        path = self._path(skill_id, tenant_id)
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception as e:
            self._logger.warning("Load skill meta failed %s: %s", skill_id, e)
            return {}

    def save(self, skill_id: str, data: dict, tenant_id: str = "default") -> None:
        path = self._path(skill_id, tenant_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def record_outcome(
        self,
        skill_id: str,
        *,
        tenant_id: str = "default",
        success: bool,
        error: Optional[str] = None,
        decay: float = 0.1,
        deprecate_threshold: float = 0.2,
    ) -> dict:
        meta = self.load(skill_id, tenant_id)
        current_rate = float(meta.get("success_rate", 1.0))
        if success:
            new_rate = min(1.0, current_rate + decay * (1.0 - current_rate))
        else:
            new_rate = max(0.0, current_rate - decay)

        anti: List[str] = list(meta.get("anti_patterns") or [])
        if error and not success and error not in anti:
            anti.append(error)

        status = meta.get("status", "active")
        if new_rate < deprecate_threshold:
            status = "deprecated"

        meta.update(
            {
                "success_rate": round(new_rate, 3),
                "last_used_at": datetime.now().isoformat(),
                "anti_patterns": anti[-20:],
                "usage_count": int(meta.get("usage_count", 0)) + 1,
                "status": status,
            }
        )
        self.save(skill_id, meta, tenant_id)
        return meta

    def set_status(
        self, skill_id: str, status: str, *, tenant_id: str = "default"
    ) -> dict:
        meta = self.load(skill_id, tenant_id)
        meta["status"] = status
        meta["status_updated_at"] = datetime.now().isoformat()
        self.save(skill_id, meta, tenant_id)
        return meta

    def merge_into_raw(
        self, skill_id: str, raw: dict, *, tenant_id: str = "default"
    ) -> dict:
        meta = self.load(skill_id, tenant_id)
        if not meta:
            return raw
        merged = dict(raw)
        for key in (
            "success_rate",
            "last_used_at",
            "anti_patterns",
            "usage_count",
            "status",
        ):
            if key in meta:
                merged[key] = meta[key]
        return merged

    def purge_tenant(self, tenant_id: str) -> int:
        base = self._tenant_dir(tenant_id)
        if not base.exists():
            return 0
        count = 0
        for path in base.glob("*.json"):
            path.unlink(missing_ok=True)
            count += 1
        return count

    def purge_user_runs_only(self) -> int:
        """Meta 按 tenant 存储，用户级 purge 不删 meta；返回 0。"""
        return 0
