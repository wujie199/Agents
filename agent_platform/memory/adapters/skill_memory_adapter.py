import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.domain.context import RequestContext
from core.ports.memory import SkillOutcome, SkillSummary
from core.ports.skills import SkillExecutionResult, SkillPort

from agent_platform.memory.adapters.skill_meta_store import SkillMetaStore
from agent_platform.memory.adapters.skill_run_audit import (
    delete_skill_runs_for_tenant,
    delete_skill_runs_for_user,
    list_skill_runs,
    record_skill_run,
)


class SkillMemoryAdapter:
    """L3 程序性记忆：搜索、执行、审计、草稿与发布。"""

    def __init__(
        self,
        skills: SkillPort,
        drafts_dir: str = "skills/drafts",
        meta_dir: str = "skills/meta",
        published_dir: Optional[str] = None,
        archive_db: Any = None,
        *,
        auto_extract_draft: bool = False,
        deprecate_threshold: float = 0.2,
        include_deprecated_in_search: bool = False,
        auto_extract_min_steps: int = 2,
    ):
        self._skills = skills
        self._drafts_dir = Path(drafts_dir)
        self._drafts_dir.mkdir(parents=True, exist_ok=True)
        self._meta = SkillMetaStore(meta_dir)
        self._published_dir = Path(
            published_dir or getattr(skills, "_skills_dir", "skills/published")
        )
        self._archive_db = archive_db
        self._auto_extract_draft = auto_extract_draft
        self._deprecate_threshold = deprecate_threshold
        self._include_deprecated = include_deprecated_in_search
        self._auto_extract_min_steps = auto_extract_min_steps
        self._logger = logging.getLogger(__name__)

    def _enrich_raw(self, skill_id: str, tenant_id: str) -> dict:
        raw = getattr(self._skills, "get_raw", lambda _id: None)(skill_id) or {}
        return self._meta.merge_into_raw(skill_id, raw, tenant_id=tenant_id)

    def _build_summary_text(self, h: dict, raw: dict) -> str:
        lines = [
            raw.get("summary") or h.get("summary") or h.get("title", h["skill_id"]),
            f"Triggers: {', '.join(h.get('triggers', [])[:3])}",
        ]
        rate = raw.get("success_rate", h.get("success_rate"))
        if rate is not None:
            lines.append(f"Success rate: {rate}")
        usage = raw.get("usage_count")
        if usage:
            lines.append(f"Usage count: {usage}")
        last = raw.get("last_used_at", h.get("last_used_at"))
        if last:
            lines.append(f"Last used: {last}")
        anti = raw.get("anti_patterns") or []
        if anti:
            lines.append(f"Anti-patterns: {'; '.join(str(a) for a in anti[:3])}")
        status = raw.get("status")
        if status and status != "active":
            lines.append(f"Status: {status}")
        return "\n".join(lines[:8])

    def search(
        self, query: str, tenant_id: str, limit: int = 3
    ) -> List[SkillSummary]:
        hits = self._skills.search(query, tenant_id, limit=limit * 2)
        summaries: List[SkillSummary] = []
        for h in hits:
            raw = self._enrich_raw(h["skill_id"], tenant_id)
            if (
                raw.get("status") == "deprecated"
                and not self._include_deprecated
            ):
                continue
            rate = raw.get("success_rate", h.get("success_rate"))
            last = raw.get("last_used_at", h.get("last_used_at"))
            summaries.append(
                SkillSummary(
                    skill_id=h["skill_id"],
                    title=h.get("title", h["skill_id"]),
                    summary=self._build_summary_text(h, raw),
                    success_rate=float(rate or 1.0),
                    last_used_at=last,
                    usage_count=int(raw.get("usage_count") or 0),
                    anti_patterns=list(raw.get("anti_patterns") or []),
                    status=str(raw.get("status") or "active"),
                )
            )
            if len(summaries) >= limit:
                break
        return summaries

    def list_skills(self, tenant_id: str) -> List[dict]:
        ids = self._skills.list_skills(tenant_id)
        rows: List[dict] = []
        for skill_id in sorted(ids):
            skill = self._skills.get(skill_id)
            raw = self._enrich_raw(skill_id, tenant_id)
            rows.append(
                {
                    "skill_id": skill_id,
                    "title": skill.title if skill else skill_id,
                    "version": skill.version if skill else "",
                    "tenant_id": skill.tenant_id if skill else None,
                    "success_rate": raw.get("success_rate", 1.0),
                    "last_used_at": raw.get("last_used_at"),
                    "usage_count": raw.get("usage_count", 0),
                    "status": raw.get("status", "active"),
                }
            )
        return rows

    def get_skill_detail(self, skill_id: str, tenant_id: str = "default") -> Optional[dict]:
        skill = self._skills.get(skill_id)
        if skill is None:
            return None
        raw = self._enrich_raw(skill_id, tenant_id)
        return {
            "skill_id": skill.skill_id,
            "title": skill.title,
            "version": skill.version,
            "tenant_id": skill.tenant_id,
            "triggers": skill.triggers,
            "trigger_regex": raw.get("trigger_regex", []),
            "required_tools": skill.required_tools,
            "acl": skill.acl,
            "summary": raw.get("summary", ""),
            "anti_patterns": raw.get("anti_patterns", []),
            "success_rate": raw.get("success_rate", 1.0),
            "last_used_at": raw.get("last_used_at"),
            "usage_count": raw.get("usage_count", 0),
            "status": raw.get("status", "active"),
            "steps": [
                {
                    "action": s.action,
                    "tool": s.tool,
                    "args_template": s.args_template,
                    "on_failure": s.on_failure,
                }
                for s in skill.steps
            ],
        }

    async def run(
        self, skill_id: str, inputs: dict, run_context: Any
    ) -> SkillExecutionResult:
        return await self._skills.run(skill_id, inputs, run_context)

    async def run_and_finalize(
        self,
        skill_id: str,
        inputs: dict,
        run_context: Any,
        context: RequestContext,
    ) -> SkillExecutionResult:
        result = await self.run(skill_id, inputs, run_context)
        await self.finalize_run(context, skill_id, inputs, result)
        return result

    async def finalize_run(
        self,
        context: RequestContext,
        skill_id: str,
        inputs: dict,
        result: SkillExecutionResult,
    ) -> dict:
        tenant_id = context.tenant_id
        self.record_outcome(
            tenant_id,
            SkillOutcome(
                skill_id=skill_id,
                success=result.success,
                steps_executed=result.steps_executed,
                error=result.error,
            ),
        )
        run_id = await record_skill_run(
            self._archive_db,
            skill_id=skill_id,
            tenant_id=tenant_id,
            user_id=context.user_id,
            session_id=context.session_id,
            trace_id=context.trace_id,
            success=result.success,
            steps_executed=result.steps_executed,
            error=result.error,
            inputs=inputs,
            outputs=result.outputs,
        )
        extra: dict = {"run_id": run_id}
        if (
            result.success
            and self._auto_extract_draft
            and result.steps_executed >= self._auto_extract_min_steps
        ):
            draft_id = self.maybe_extract_draft_from_skill(
                tenant_id, skill_id, suffix="_auto"
            )
            if draft_id:
                extra["auto_draft_id"] = draft_id
        return extra

    async def on_session_end(self, context: RequestContext) -> dict:
        """会话结束时：对本轮成功 skill runs 补做 auto_extract（配置开启时）。"""
        if not self._auto_extract_draft:
            return {"enabled": False, "drafts": []}
        runs = await self.list_skill_runs(
            context.tenant_id, user_id=context.user_id, limit=100
        )
        session_runs = [
            r for r in runs if r.get("session_id") == context.session_id
        ]
        drafts: List[dict] = []
        seen: set[str] = set()
        for run in session_runs:
            if not run.get("success"):
                continue
            steps = int(run.get("steps_executed") or 0)
            if steps < self._auto_extract_min_steps:
                continue
            skill_id = str(run.get("skill_id") or "")
            if not skill_id or skill_id in seen:
                continue
            seen.add(skill_id)
            draft_id = self.maybe_extract_draft_from_skill(
                context.tenant_id, skill_id, suffix="_auto"
            )
            if draft_id:
                drafts.append({"skill_id": skill_id, "draft_id": draft_id})
        return {
            "enabled": True,
            "runs_scanned": len(session_runs),
            "drafts": drafts,
        }

    def record_outcome(
        self,
        tenant_id: str,
        outcome: SkillOutcome,
        decay: float = 0.1,
    ) -> None:
        if self._skills.get(outcome.skill_id) is None:
            return
        self._meta.record_outcome(
            outcome.skill_id,
            tenant_id=tenant_id,
            success=outcome.success,
            error=outcome.error,
            decay=decay,
            deprecate_threshold=self._deprecate_threshold,
        )

    def deprecate_skill(self, tenant_id: str, skill_id: str) -> dict:
        if self._skills.get(skill_id) is None:
            return {"success": False, "reason": "skill_not_found"}
        meta = self._meta.set_status(skill_id, "deprecated", tenant_id=tenant_id)
        return {"success": True, "skill_id": skill_id, "meta": meta}

    def activate_skill(self, tenant_id: str, skill_id: str) -> dict:
        if self._skills.get(skill_id) is None:
            return {"success": False, "reason": "skill_not_found"}
        meta = self._meta.set_status(skill_id, "active", tenant_id=tenant_id)
        return {"success": True, "skill_id": skill_id, "meta": meta}

    def maybe_extract_draft_from_skill(
        self,
        tenant_id: str,
        skill_id: str,
        *,
        suffix: str = "_draft",
    ) -> Optional[str]:
        skill = self._skills.get(skill_id)
        if skill is None:
            return None
        draft_id = f"{skill_id}{suffix}"[:48]
        steps = [
            {
                "action": s.action,
                "tool": s.tool,
                "args_template": s.args_template,
                "on_failure": s.on_failure,
            }
            for s in skill.steps
        ]
        return self.extract_draft(
            tenant_id,
            f"{skill.title} (auto)",
            list(skill.triggers),
            steps,
            skill_id=draft_id,
        )

    def purge_l3_for_user(self, tenant_id: str, user_id: str) -> dict:
        return {
            "skill_runs_deleted": 0,
            "drafts_removed": 0,
            "meta_files_removed": 0,
        }

    async def purge_l3_for_user_async(self, tenant_id: str, user_id: str) -> dict:
        runs = await delete_skill_runs_for_user(
            self._archive_db, tenant_id, user_id
        )
        return {
            "skill_runs_deleted": runs,
            "drafts_removed": 0,
            "meta_files_removed": 0,
            "note": "purge-user 仅删除 skill_runs；租户 drafts 请用 purge-tenant-l3",
        }

    async def list_skill_runs(
        self,
        tenant_id: str,
        *,
        user_id: Optional[str] = None,
        skill_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[dict]:
        return await list_skill_runs(
            self._archive_db,
            tenant_id=tenant_id,
            user_id=user_id,
            skill_id=skill_id,
            limit=limit,
            offset=offset,
        )

    async def purge_l3_for_tenant_async(
        self, tenant_id: str, *, delete_runs: bool = True
    ) -> dict:
        runs = 0
        if delete_runs:
            runs = await delete_skill_runs_for_tenant(
                self._archive_db, tenant_id
            )
        drafts = self.purge_drafts_for_tenant(tenant_id)
        meta = self._meta.purge_tenant(tenant_id)
        return {
            "skill_runs_deleted": runs,
            "drafts_removed": drafts,
            "meta_files_removed": meta,
        }

    def purge_drafts_for_tenant(self, tenant_id: str) -> int:
        base = self._drafts_dir / tenant_id
        if not base.exists():
            return 0
        count = sum(1 for _ in base.glob("*/skill.yaml"))
        shutil.rmtree(base, ignore_errors=True)
        return count

    def extract_draft(
        self,
        tenant_id: str,
        title: str,
        triggers: List[str],
        steps: List[Dict[str, Any]],
        skill_id: Optional[str] = None,
        trigger_regex: Optional[List[str]] = None,
    ) -> str:
        draft_id = skill_id or re.sub(r"[^\w\-]", "_", title.lower())[:32]
        draft_path = self._drafts_dir / tenant_id / draft_id / "skill.yaml"
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "skill_id": draft_id,
            "version": "0.1.0-draft",
            "title": title,
            "triggers": triggers,
            "trigger_regex": trigger_regex or [],
            "steps": steps,
            "tenant_id": tenant_id,
            "status": "draft",
            "acl": ["user", "cli", "test"],
            "created_at": datetime.now().isoformat(),
        }
        with open(draft_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
        self._logger.info("Skill draft saved: %s", draft_path)
        return draft_id

    def list_drafts(self, tenant_id: str) -> List[dict]:
        base = self._drafts_dir / tenant_id
        if not base.exists():
            return []
        rows: List[dict] = []
        for skill_file in base.glob("*/skill.yaml"):
            with open(skill_file, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            rows.append(
                {
                    "skill_id": data.get("skill_id", skill_file.parent.name),
                    "title": data.get("title", ""),
                    "version": data.get("version", ""),
                    "status": data.get("status", "draft"),
                }
            )
        return sorted(rows, key=lambda r: r["skill_id"])

    def publish_skill(
        self,
        tenant_id: str,
        skill_id: str,
        *,
        remove_draft: bool = True,
    ) -> dict:
        draft_path = self._drafts_dir / tenant_id / skill_id / "skill.yaml"
        if not draft_path.exists():
            return {"success": False, "reason": "draft_not_found"}

        with open(draft_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        data["status"] = "published"
        version = str(data.get("version", "0.1.0")).replace("-draft", "")
        if version and not version.endswith(".0"):
            version = f"{version}.0" if "." not in version else version
        data["version"] = version
        data.pop("created_at", None)
        data["published_at"] = datetime.now().isoformat()
        data.setdefault("acl", ["user", "cli", "test"])

        dest_dir = self._published_dir / skill_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / "skill.yaml"
        with open(dest_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

        if remove_draft:
            shutil.rmtree(draft_path.parent, ignore_errors=True)

        if hasattr(self._skills, "reload"):
            self._skills.reload()

        self._logger.info("Published skill %s -> %s", skill_id, dest_file)
        return {"success": True, "skill_id": skill_id, "path": str(dest_file)}

    def sync_from_source(
        self, source_dir: str, *, remove_missing: bool = False
    ) -> dict:
        src = Path(source_dir)
        if not src.exists():
            return {"success": False, "reason": "source_not_found"}
        copied = 0
        for skill_file in src.glob("*/skill.yaml"):
            dest = self._published_dir / skill_file.parent.name / "skill.yaml"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(skill_file, dest)
            copied += 1
        if remove_missing:
            src_ids = {p.parent.name for p in src.glob("*/skill.yaml")}
            for existing in self._published_dir.glob("*/skill.yaml"):
                if existing.parent.name not in src_ids:
                    shutil.rmtree(existing.parent, ignore_errors=True)
        if hasattr(self._skills, "reload"):
            self._skills.reload()
        return {"success": True, "copied": copied}
