# -*- coding: utf-8 -*-
"""DeepAgent ↔ RunContext 桥接适配。

隔离 DeepAgents API 变更，提供统一接口。
明确边界：
- DeepAgents Persistent Memory 只存规划状态（TodoList）
- 业务记忆只走内层 MemoryPort（禁止双写）
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.composition.run_context import RunContext

from app.runtime.adapters.deepagents.config import DeepAgentConfig
from app.runtime.adapters.deepagents.subagent_bridge import InnerSubAgentBridge

logger = logging.getLogger(__name__)


class DeepAgentAdapter:
    """适配层：将 DeepAgents harness 与当前 RunContext 体系桥接。"""

    def __init__(
        self,
        ctx: RunContext,
        da_config: DeepAgentConfig,
        inner_bridge: Optional[InnerSubAgentBridge] = None,
    ) -> None:
        self._ctx = ctx
        self._config = da_config
        self._inner_bridge = inner_bridge
        self._agent: Any = None

    async def ensure_agent(self) -> Any:
        """延迟创建 DeepAgent（首次调用时组装）。"""
        if self._agent is not None:
            return self._agent

        from app.runtime.adapters.deepagents.harness import create_planning_agent

        self._agent = await create_planning_agent(
            self._ctx,
            self._config,
            inner_bridge=self._inner_bridge,
        )
        return self._agent

    async def invoke(self, user_input: str) -> Dict[str, Any]:
        """执行 DeepAgent 规划 + 子Agent 调度。"""
        agent = await self.ensure_agent()
        try:
            result = await agent.ainvoke({"messages": user_input})
            return self._extract_result(result)
        except Exception as e:
            logger.error("DeepAgent invoke failed: %s", e, exc_info=True)
            return {"assistant_text": "", "error": str(e)}

    @staticmethod
    def _extract_result(result: Any) -> Dict[str, Any]:
        """从 DeepAgent 返回值提取结构化结果。"""
        if isinstance(result, dict):
            messages = result.get("messages") or []
            last_text = ""
            for msg in reversed(messages):
                content = getattr(msg, "content", None) or ""
                if content and isinstance(content, str):
                    last_text = content
                    break
            return {
                "assistant_text": last_text,
                "todos": result.get("todos"),
                "subagent_results": result.get("subagent_results"),
            }
        return {"assistant_text": str(result) if result else ""}

    async def get_todos(self) -> List[Dict[str, Any]]:
        """获取当前规划状态（TodoList）。"""
        # DeepAgents Persistent Memory 中只含 TodoList
        agent = await self.ensure_agent()
        try:
            state = await agent.aget_state({})
            return state.values.get("todos") or []
        except Exception:
            return []
