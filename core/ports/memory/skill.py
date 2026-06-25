"""L3 技能记忆端口 — 技能搜索、执行、发布、管理。"""

from typing import Protocol, Optional, List, Any

from core.domain.context import RequestContext
from core.ports.memory.dtos import SkillSummary, SkillOutcome


class SkillMemoryPort(Protocol):
    """L3 技能记忆契约：技能搜索/执行/发布/管理"""

    async def skill_search(
        self,
        query: str,
        context: RequestContext,
        limit: int = 3,
    ) -> List[SkillSummary]:
        ...

    async def run_skill(
        self,
        skill_id: str,
        inputs: dict,
        context: RequestContext,
        run_context: Any,
    ) -> Any:
        ...

    async def list_skills(self, context: RequestContext) -> List[dict]:
        ...

    async def get_skill(
        self, skill_id: str, context: RequestContext
    ) -> Optional[dict]:
        ...

    async def extract_skill_draft(
        self,
        context: RequestContext,
        title: str,
        triggers: List[str],
        steps: List[dict],
        skill_id: Optional[str] = None,
    ) -> str:
        ...

    async def list_skill_drafts(self, context: RequestContext) -> List[dict]:
        ...

    async def publish_skill(
        self,
        context: RequestContext,
        skill_id: str,
        *,
        remove_draft: bool = True,
    ) -> dict:
        ...

    async def deprecate_skill(
        self, context: RequestContext, skill_id: str
    ) -> dict:
        ...

    async def activate_skill(
        self, context: RequestContext, skill_id: str
    ) -> dict:
        ...

    async def sync_skills_from(
        self,
        source_dir: str,
        *,
        remove_missing: bool = False,
    ) -> dict:
        ...

    async def list_skill_runs(
        self,
        context: RequestContext,
        *,
        skill_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[dict]:
        ...

    async def purge_tenant_l3(
        self,
        tenant_id: str,
        *,
        delete_runs: bool = True,
    ) -> dict:
        ...

    async def record_skill_outcome(
        self,
        context: RequestContext,
        outcome: SkillOutcome,
    ) -> None:
        ...
