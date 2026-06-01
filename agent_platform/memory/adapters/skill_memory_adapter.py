import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.domain.context import RequestContext
from core.ports.external_memory import Entity, Fact
from core.ports.memory import SkillOutcome, SkillSummary
from core.ports.skills import SkillExecutionResult, SkillPort


class SkillMemoryAdapter:
    """L3 程序性记忆：搜索摘要、执行、结果记录、草稿提取。"""

    def __init__(
        self,
        skills: SkillPort,
        drafts_dir: str = "skills/drafts",
    ):
        self._skills = skills
        self._drafts_dir = Path(drafts_dir)
        self._drafts_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)

    def search(
        self, query: str, tenant_id: str, limit: int = 3
    ) -> List[SkillSummary]:
        hits = self._skills.search(query, tenant_id, limit=limit)
        summaries: List[SkillSummary] = []
        for h in hits:
            lines = [
                h.get("summary") or h.get("title", h["skill_id"]),
                f"Triggers: {', '.join(h.get('triggers', [])[:3])}",
            ]
            rate = h.get("success_rate")
            if rate is not None:
                lines.append(f"Success rate: {rate}")
            last = h.get("last_used_at")
            if last:
                lines.append(f"Last used: {last}")
            summaries.append(
                SkillSummary(
                    skill_id=h["skill_id"],
                    title=h.get("title", h["skill_id"]),
                    summary="\n".join(lines[:5]),
                    success_rate=float(rate or 1.0),
                    last_used_at=last,
                )
            )
        return summaries

    def run(
        self, skill_id: str, inputs: dict, run_context: Any
    ) -> SkillExecutionResult:
        return self._skills.run(skill_id, inputs, run_context)

    def record_outcome(
        self,
        tenant_id: str,
        outcome: SkillOutcome,
        decay: float = 0.1,
    ) -> None:
        raw = getattr(self._skills, "get_raw", lambda _id: None)(outcome.skill_id)
        if raw is None:
            return
        skill_dir = Path(getattr(self._skills, "_skills_dir", "skills/published"))
        skill_file = skill_dir / outcome.skill_id / "skill.yaml"
        if not skill_file.exists():
            return

        current_rate = float(raw.get("success_rate", 1.0))
        if outcome.success:
            new_rate = min(1.0, current_rate + decay * (1.0 - current_rate))
        else:
            new_rate = max(0.0, current_rate - decay)

        raw["success_rate"] = round(new_rate, 3)
        raw["last_used_at"] = datetime.now().isoformat()
        if outcome.error and not outcome.success:
            anti = raw.setdefault("anti_patterns", [])
            if outcome.error not in anti:
                anti.append(outcome.error)

        with open(skill_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)

        if hasattr(self._skills, "reload"):
            self._skills.reload()

    def extract_draft(
        self,
        tenant_id: str,
        title: str,
        triggers: List[str],
        steps: List[Dict[str, Any]],
        skill_id: Optional[str] = None,
    ) -> str:
        draft_id = skill_id or title.replace(" ", "_").lower()[:32]
        draft_path = self._drafts_dir / tenant_id / draft_id / "skill.yaml"
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "skill_id": draft_id,
            "version": "0.1.0-draft",
            "title": title,
            "triggers": triggers,
            "steps": steps,
            "tenant_id": tenant_id,
            "status": "draft",
            "created_at": datetime.now().isoformat(),
        }
        with open(draft_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
        self._logger.info("Skill draft saved: %s", draft_path)
        return draft_id
