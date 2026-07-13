"""统一工具路由：Memory / MCP / Native 工具 → 对应 Port。"""

from __future__ import annotations

from typing import Any, Dict, Set

import time

from core.composition.run_context import RunContext
from core.ports.observability import Layer

_MEMORY_TOOLS = frozenset(
    {
        "session_search",
        "session_search_detail",
        "skill_search",
        "run_skill",
        "remember_user_fact",
        "memory",
    }
)


def collect_available_tool_names(ctx: RunContext) -> Set[str]:
    """供 Skill 执行前 required_tools 校验。"""
    names: Set[str] = set(_MEMORY_TOOLS)
    if ctx.tools is not None and hasattr(ctx.tools, "list_tools"):
        names.update(ctx.tools.list_tools())
    if ctx.mcp is not None and hasattr(ctx.mcp, "list_servers"):
        for server_id in ctx.mcp.list_servers():
            names.add(f"mcp.{server_id}.*")
    return names


def _tool_available(tool_name: str, available: Set[str]) -> bool:
    if tool_name in available:
        return True
    if tool_name.startswith("mcp.") and ctx_mcp_wildcards(available, tool_name):
        return True
    return False


def ctx_mcp_wildcards(available: Set[str], tool_name: str) -> bool:
    parts = tool_name.split(".", 2)
    if len(parts) < 3:
        return False
    return f"mcp.{parts[1]}.*" in available


def validate_required_tools(
    required: list[str], ctx: RunContext
) -> list[str]:
    available = collect_available_tool_names(ctx)
    missing = []
    for name in required:
        if name.startswith("mcp."):
            parts = name.split(".", 2)
            if len(parts) < 3:
                missing.append(name)
                continue
            server_id = parts[1]
            if ctx.mcp is None or (
                hasattr(ctx.mcp, "list_servers")
                and server_id not in ctx.mcp.list_servers()
            ):
                missing.append(name)
            continue
        if name not in available:
            missing.append(name)
    return missing


async def invoke_tool(
    ctx: RunContext,
    tool_name: str,
    args: Dict[str, Any],
) -> Any:
    from app.agents.observability.graph_metrics import record_tool_call
    from app.agents.observability.instrument import span_ctx

    span_attrs: Dict[str, Any] = {"tool_name": tool_name}
    async with span_ctx(ctx, "agent.invoke_tool", Layer.AGENT, span_attrs):
        t0 = time.perf_counter()
        success = True
        try:
            return await _invoke_tool_impl(ctx, tool_name, args)
        except Exception:
            success = False
            raise
        finally:
            span_attrs["success"] = success
            record_tool_call(
                ctx,
                (time.perf_counter() - t0) * 1000,
                tool_name=tool_name,
                success=success,
            )


async def _invoke_tool_impl(
    ctx: RunContext,
    tool_name: str,
    args: Dict[str, Any],
) -> Any:
    if tool_name in _MEMORY_TOOLS:
        memory = ctx.require_memory()
        if tool_name == "session_search":
            return await memory.session_search(
                args["query"],
                ctx.request,
                limit=int(args.get("limit", 5)),
                scope=args.get("scope", "session"),
                mode=args.get("mode"),
                sort=args.get("sort", "newest"),
                around_message_id=str(args.get("around_message_id") or ""),
                session_link=str(args.get("session_link") or ""),
            )
        if tool_name == "session_search_detail":
            detail = await memory.session_search_detail(
                args["query"],
                ctx.request,
                limit=int(args.get("limit", 5)),
                scope=args.get("scope", "session"),
            )
            return detail.to_dict()
        if tool_name == "skill_search":
            hits = await memory.skill_search(
                args["query"], ctx.request, limit=int(args.get("limit", 3))
            )
            return [
                {
                    "skill_id": h.skill_id,
                    "title": h.title,
                    "summary": h.summary,
                    "success_rate": h.success_rate,
                    "last_used_at": h.last_used_at,
                    "usage_count": getattr(h, "usage_count", 0),
                    "anti_patterns": getattr(h, "anti_patterns", []),
                    "status": getattr(h, "status", "active"),
                }
                for h in hits
            ]
        if tool_name == "run_skill":
            result = await memory.run_skill(
                args["skill_id"],
                args.get("inputs") or {},
                ctx.request,
                ctx,
            )
            return {
                "skill_id": result.skill_id,
                "success": result.success,
                "steps_executed": result.steps_executed,
                "outputs": result.outputs,
                "error": result.error,
            }
        if tool_name == "remember_user_fact":
            from core.ports.memory import MemoryDelta

            from app.agents.memory.memory_graph_state import record_l1_write

            require_hitl = bool(args.get("require_hitl", True))
            key = str(args["key"])
            value = str(args["value"])
            await memory.update_prompt_memory(
                ctx.request,
                MemoryDelta(key=key, value=value, source="user"),
                require_hitl=require_hitl,
            )
            record_l1_write(
                ctx,
                key=key,
                value=value,
                source="user",
                require_hitl=require_hitl,
            )
            return {"ok": True, "key": key, "pending": require_hitl}
        if tool_name == "memory":
            from app.agents.memory.memory_graph_state import record_memory_tool_result

            invoke = getattr(memory, "invoke_memory_tool", None)
            if invoke is None:
                return {"success": False, "error": "L1 memory tool not available."}
            ops = args.get("operations")
            action = str(args.get("action", ""))
            target = str(args.get("target", "memory"))
            content = args.get("content")
            result = invoke(
                ctx.request,
                action=action,
                target=target,
                content=content,
                old_text=args.get("old_text"),
                operations=ops if isinstance(ops, list) else None,
            )
            if isinstance(result, dict):
                record_memory_tool_result(
                    ctx,
                    result,
                    action=action,
                    target=target,
                    content=str(content) if content else None,
                )
            return result

    if tool_name.startswith("mcp."):
        parts = tool_name.split(".", 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid MCP tool name: {tool_name}")
        server_id, mcp_tool = parts[1], parts[2]
        if ctx.mcp is None:
            raise RuntimeError("MCPPort not configured")
        result = await ctx.mcp.call_tool(server_id, mcp_tool, args)
        if not result.success:
            raise RuntimeError(result.error or f"MCP call failed: {tool_name}")
        return result.output

    if tool_name.startswith("skill."):
        nested_id = tool_name.split(".", 1)[1]
        memory = ctx.require_memory()
        nested_inputs = args.get("inputs", args)
        nested = await memory.run_skill(
            nested_id, nested_inputs, ctx.request, ctx
        )
        return {
            "skill_id": nested.skill_id,
            "success": nested.success,
            "outputs": nested.outputs,
            "error": nested.error,
        }

    tools = ctx.require_tools()
    return await tools.invoke(tool_name, args, ctx.request)
