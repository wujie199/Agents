# -*- coding: utf-8 -*-
"""DeepAgents harness 组装：创建外层规划 Agent。

核心逻辑：
1. 当前系统编译图作为 chat-worker 子Agent
2. 额外子Agent（research-agent, report-writer 等）从配置读取
3. 组装 create_deep_agent()，注入 Middleware 和 checkpointer

职责划分：
- DeepAgents TodoListMiddleware → 任务分解 + 依赖排序
- DeepAgents SubAgentMiddleware → 子Agent 委托 + 上下文隔离
- DeepAgents FilesystemMiddleware → 大结果自动落盘 + 上下文缓冲
- DeepAgents PatchToolCallsMiddleware → 中断恢复
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.composition.run_context import RunContext

from app.runtime.adapters.deepagents.config import DeepAgentConfig
from app.runtime.adapters.deepagents.subagent_bridge import InnerSubAgentBridge

logger = logging.getLogger(__name__)


async def create_planning_agent(
    ctx: RunContext,
    da_config: DeepAgentConfig,
    *,
    inner_bridge: Optional[InnerSubAgentBridge] = None,
    checkpointer: Any = None,
) -> Any:
    """组装 DeepAgent 外层规划 harness。

    Args:
        ctx: RunContext（当前系统运行上下文）
        da_config: DeepAgent 配置
        inner_bridge: 内层子Agent 桥接（chat-worker）
        checkpointer: 可选的 LangGraph checkpointer

    Returns:
        CompiledStateGraph: DeepAgent 编译后的图
    """
    try:
        from deepagents import create_deep_agent
    except ImportError as e:
        raise ImportError(
            "deepagents 未安装。请执行: pip install 'agents[planning]'"
        ) from e

    # ① 组装子Agent 列表
    subagents = _build_subagents(ctx, da_config, inner_bridge)

    # ② 组装工具列表（顶层 custom tools）
    tools = _resolve_tools(ctx, da_config)

    # ③ 解析 checkpointer
    cp = checkpointer
    if cp is None:
        try:
            from app.runtime.adapters.langgraph.checkpointer import (
                resolve_chat_checkpointer_async,
            )

            cp = await resolve_chat_checkpointer_async(ctx)
        except Exception:
            logger.warning("No checkpointer for DeepAgent, using in-memory")

    # ④ 获取规划 LLM
    model_name = _resolve_model_name(ctx, da_config)

    # ⑤ 组装 DeepAgent
    agent_kwargs: Dict[str, Any] = {
        "model": model_name,
        "tools": tools,
        "subagents": subagents,
        "system_prompt": da_config.system_prompt,
    }

    # HITL 中断配置
    if da_config.interrupt_on:
        agent_kwargs["interrupt_on"] = da_config.interrupt_on

    # checkpointer
    if cp is not None:
        agent_kwargs["checkpointer"] = cp

    agent = create_deep_agent(**agent_kwargs)
    logger.info(
        "DeepAgent harness created: model=%s, subagents=%d, tools=%d",
        model_name,
        len(subagents),
        len(tools),
    )
    return agent


def _build_subagents(
    ctx: RunContext,
    da_config: DeepAgentConfig,
    inner_bridge: Optional[InnerSubAgentBridge],
) -> List[Any]:
    """组装子Agent 列表。"""
    subagents: List[Any] = []

    # chat-worker：内层图作为子Agent
    if inner_bridge is not None:
        try:
            from deepagents import CompiledSubAgent

            subagents.append(
                CompiledSubAgent(
                    name=inner_bridge.name,
                    graph=inner_bridge._compiled,
                    description=inner_bridge.description,
                )
            )
        except (ImportError, AttributeError):
            # deepagents API 可能变化，降级为 dict spec
            subagents.append(inner_bridge.to_subagent_spec())

    # 额外子Agent（从配置读取）
    for spec in da_config.subagents:
        if isinstance(spec, dict) and spec.get("name"):
            subagents.append(spec)

    return subagents


def _resolve_tools(ctx: RunContext, da_config: DeepAgentConfig) -> List[Any]:
    """解析顶层自定义工具。"""
    if not da_config.custom_tool_names:
        return []
    # 工具按名称从 RunContext 或全局注册表解析
    # 当前阶段返回空列表，后续按需接入
    return []


def _resolve_model_name(ctx: RunContext, da_config: DeepAgentConfig) -> str:
    """解析规划 LLM 名称。"""
    # 尝试从 models 配置获取
    try:
        models = ctx.models
        if models is not None:
            model = models.get_model(da_config.planning_model_role)
            if hasattr(model, "model_name"):
                return str(model.model_name)
    except Exception:
        pass
    # 默认
    return "openai:gpt-4o"
