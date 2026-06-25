# -*- coding: utf-8 -*-
"""子Agent 桥接：将当前系统的 CompiledStateGraph 转换为 DeepAgents CompiledSubAgent。

内层图作为 chat-worker 子Agent 注入外层 DeepAgents harness。
严格过滤：只传 messages + user_input 进子Agent，不传内部状态。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.composition.run_context import RunContext


class InnerSubAgentBridge:
    """桥接内层 LangGraph CompiledStateGraph → DeepAgents 子Agent。

    封装调用逻辑，确保：
    1. 只传必要输入（user_input / messages）
    2. 不泄漏内层状态（memory_snapshot_hash 等）
    3. 返回值提取 assistant_text
    """

    def __init__(
        self,
        compiled_graph: Any,
        runtime: Any,
        name: str = "chat-worker",
        description: str = "执行单轮对话：意图路由→L1+RAG检索→LLM推理→记忆持久化",
    ) -> None:
        self._compiled = compiled_graph
        self._runtime = runtime
        self.name = name
        self.description = description

    async def invoke(
        self,
        ctx: RunContext,
        task_input: str,
        *,
        enable_rag: bool = True,
        chat_cfg: Any = None,
    ) -> Dict[str, Any]:
        """执行子Agent：调用内层图，返回结构化结果。"""
        input_state = {"user_input": task_input.strip()}
        final = await self._runtime.ainvoke(
            self._compiled,
            input_state,
            ctx,
            enable_rag=enable_rag,
            chat_cfg=chat_cfg,
        )
        # 严格过滤：只返回外层需要的字段
        return {
            "assistant_text": final.get("assistant_text") or "",
            "evidence_count": int(final.get("evidence_count") or 0),
            "rag_empty": bool(final.get("rag_empty", True)),
        }

    def to_subagent_spec(self) -> Dict[str, Any]:
        """转为 DeepAgents subagent 配置格式。"""
        return {
            "name": self.name,
            "description": self.description,
            # graph 和 invoke 由 harness 组装时处理
        }
