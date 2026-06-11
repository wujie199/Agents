# -*- coding: utf-8 -*-
"""LangGraph Agent 记忆工具（Path B）：L1/L2/L3/L4 全量。"""

from __future__ import annotations

import json
from typing import Any, List

from langchain_core.tools import StructuredTool

from core.composition.run_context import RunContext
from core.ports.memory import MemoryDelta

from app.agents.chat_config import ChatAgentConfig
from app.agents.context_builder import validate_l1_key
from app.agents.memory_metrics import record_session_search


def build_memory_tools(
    ctx: RunContext, chat_cfg: ChatAgentConfig
) -> List[StructuredTool]:
    if not chat_cfg.enable_memory_tools:
        return []

    async def session_search(
        query: str,
        limit: int = 5,
        scope: str = "session",
    ) -> str:
        """搜索会话历史。scope=session 本会话；scope=user 跨会话。"""
        memory = ctx.require_memory()
        use_scope = "user" if str(scope).lower() == "user" else "session"
        text = await memory.session_search(
            query,
            ctx.request,
            limit=max(1, min(int(limit), 20)),
            scope=use_scope,  # type: ignore[arg-type]
        )
        record_session_search(ctx, hit=bool(text and text.strip()))
        return text or "（未找到相关历史）"

    async def remember_user_fact(key: str, value: str) -> str:
        """将用户事实或偏好写入长期记忆 USER profile。key 如：称呼、语言、输出格式。"""
        safe_key = validate_l1_key(key, chat_cfg.l1_extract_allowed_keys)
        val = (value or "").strip()
        if not val:
            return "错误：value 不能为空"
        memory = ctx.require_memory()
        await memory.update_prompt_memory(
            ctx.request,
            MemoryDelta(key=safe_key, value=val, source="user"),
            require_hitl=chat_cfg.remember_require_hitl,
        )
        if chat_cfg.remember_require_hitl:
            return (
                f"已加入待确认记忆：{safe_key}={val}。"
                "输入 /pending 查看，/confirm 立即写入 L1；"
                "或会话结束时自动 finalize。"
            )
        return f"已记住：{safe_key}={val}"

    tools: List[StructuredTool] = [
        StructuredTool.from_function(
            coroutine=session_search,
            name="session_search",
            description=(
                "搜索历史对话片段。scope=session 查本会话；scope=user 查该用户所有会话。"
                "回忆类问题、跨会话追溯时使用。"
            ),
        ),
        StructuredTool.from_function(
            coroutine=remember_user_fact,
            name="remember_user_fact",
            description=(
                "记住用户的长期偏好或事实到 USER 记忆。"
                "当用户说「记住」「以后都」「叫我…」时使用。"
            ),
        ),
    ]

    if chat_cfg.enable_skill_tools:

        async def skill_search(query: str, limit: int = 3) -> str:
            """搜索可复用技能（L3），返回 skill_id 与摘要。"""
            memory = ctx.require_memory()
            hits = await memory.skill_search(
                query,
                ctx.request,
                limit=max(1, min(int(limit), 10)),
            )
            rows = [
                {
                    "skill_id": h.skill_id,
                    "title": h.title,
                    "summary": h.summary,
                    "success_rate": h.success_rate,
                    "status": h.status,
                }
                for h in hits
            ]
            return json.dumps(rows, ensure_ascii=False)

        async def run_skill(skill_id: str, inputs: str = "{}") -> str:
            """执行已发布技能。inputs 为 JSON 对象字符串。"""
            memory = ctx.require_memory()
            try:
                payload = json.loads(inputs or "{}")
                if not isinstance(payload, dict):
                    payload = {}
            except json.JSONDecodeError:
                return "错误：inputs 必须是 JSON 对象"
            result = await memory.run_skill(
                skill_id, payload, ctx.request, ctx
            )
            return json.dumps(
                {
                    "skill_id": result.skill_id,
                    "success": result.success,
                    "steps_executed": result.steps_executed,
                    "outputs": getattr(result, "outputs", None),
                    "error": result.error,
                },
                ensure_ascii=False,
            )

        tools.extend(
            [
                StructuredTool.from_function(
                    coroutine=skill_search,
                    name="skill_search",
                    description=(
                        "搜索 L3 技能库。多步任务、报告生成、JSON 读写等场景先搜索再 run_skill。"
                    ),
                ),
                StructuredTool.from_function(
                    coroutine=run_skill,
                    name="run_skill",
                    description="执行指定 skill_id 的技能，inputs 传 JSON 参数字符串。",
                ),
            ]
        )

    if chat_cfg.enable_l4_tools:

        async def resolve_entity(mention: str) -> str:
            """解析实体别名（L4），如人名→canonical_id。"""
            memory = ctx.require_memory()
            resolver = getattr(memory, "resolve_entity", None)
            if resolver is None:
                return "（L4 未配置）"
            entity = await resolver(mention, ctx.request)
            if entity is None:
                return f"未解析到实体：{mention}"
            return json.dumps(
                {
                    "mention": entity.mention,
                    "canonical_id": entity.canonical_id,
                    "display_name": entity.display_name,
                },
                ensure_ascii=False,
            )

        async def fetch_profile_facts() -> str:
            """读取当前用户 L4 外部画像 facts（不触发 L1 合并）。"""
            memory = ctx.require_memory()
            fetcher = getattr(memory, "fetch_profile_facts", None)
            if fetcher is None:
                return "[]"
            facts = await fetcher(ctx.request.tenant_id, ctx.request.user_id)
            rows = [
                {"key": f.key, "value": f.value, "source": getattr(f, "source", "")}
                for f in facts
            ]
            return json.dumps(rows, ensure_ascii=False)

        tools.extend(
            [
                StructuredTool.from_function(
                    coroutine=resolve_entity,
                    name="resolve_entity",
                    description="解析用户提到的实体别名（CRM/LDAP 画像）。",
                ),
                StructuredTool.from_function(
                    coroutine=fetch_profile_facts,
                    name="fetch_profile_facts",
                    description="读取当前用户外部画像结构化 facts。",
                ),
            ]
        )

    return tools
