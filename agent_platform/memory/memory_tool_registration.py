"""将 MemoryPort 工具注册到 ToolPort。"""

from __future__ import annotations

from typing import Any, List

from core.domain.context import RequestContext
from core.composition.run_context import RunContext


def register_memory_tools(tools: Any, memory: Any) -> None:
    async def session_search(
        query: str,
        context: RequestContext,
        limit: int = 5,
        scope: str = "session",
    ) -> str:
        return await memory.session_search(
            query, context, limit=limit, scope=scope
        )

    async def session_search_detail(
        query: str,
        context: RequestContext,
        limit: int = 5,
        scope: str = "session",
    ) -> dict:
        result = await memory.session_search_detail(
            query, context, limit=limit, scope=scope
        )
        return result.to_dict()

    async def skill_search(
        query: str,
        context: RequestContext,
        limit: int = 3,
    ) -> List[dict]:
        hits = await memory.skill_search(query, context, limit=limit)
        return [
            {
                "skill_id": h.skill_id,
                "title": h.title,
                "summary": h.summary,
                "success_rate": h.success_rate,
                "last_used_at": h.last_used_at,
                "usage_count": h.usage_count,
                "anti_patterns": h.anti_patterns,
                "status": h.status,
            }
            for h in hits
        ]

    async def run_skill(
        skill_id: str,
        context: RequestContext,
        inputs: dict | None = None,
    ) -> dict:
        run_ctx = RunContext(request=context, memory=memory, tools=tools)
        result = await memory.run_skill(
            skill_id, inputs or {}, context, run_ctx
        )
        return {
            "skill_id": result.skill_id,
            "success": result.success,
            "steps_executed": result.steps_executed,
            "outputs": result.outputs,
            "error": result.error,
        }

    async def resolve_entity(
        mention: str,
        context: RequestContext,
    ) -> dict | None:
        entity = await memory.resolve_entity(mention, context)
        if entity is None:
            return None
        return {
            "mention": entity.mention,
            "canonical_id": entity.canonical_id,
            "display_name": entity.display_name,
        }

    async def fetch_profile_facts(
        context: RequestContext,
    ) -> List[dict]:
        return await memory.fetch_profile_facts(
            context.tenant_id, context.user_id
        )

    tools.register_tool("session_search", session_search, acl=["user"])
    tools.register_tool(
        "session_search_detail", session_search_detail, acl=["user"]
    )
    tools.register_tool("skill_search", skill_search, acl=["user"])
    tools.register_tool("run_skill", run_skill, acl=["user", "cli", "test"])
    tools.register_tool("resolve_entity", resolve_entity, acl=["user", "cli"])
    tools.register_tool(
        "fetch_profile_facts", fetch_profile_facts, acl=["user", "cli"]
    )
