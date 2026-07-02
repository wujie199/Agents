# -*- coding: utf-8 -*-
"""LangGraph 编译图 invoke / astream 封装。"""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from core.composition.run_context import RunContext


class LangGraphRuntime:
    def __init__(self, checkpointer: Any = None) -> None:
        self._checkpointer = checkpointer

    def compile(self, graph: Any) -> Any:
        if self._checkpointer is not None:
            return graph.compile(checkpointer=self._checkpointer)
        return graph.compile()

    def build_config(
        self,
        ctx: RunContext,
        *,
        enable_rag: bool = True,
        chat_cfg: Any = None,
        extra: Optional[dict] = None,
    ) -> dict:
        configurable = {
            "thread_id": ctx.request.session_id,
            "run_ctx": ctx,
            "enable_rag": enable_rag,
        }
        trace_id = getattr(ctx.request, "trace_id", None)
        if trace_id:
            configurable["trace_id"] = trace_id
        if chat_cfg is not None:
            configurable["chat_cfg"] = chat_cfg
        if extra:
            configurable.update(extra)
        return {"configurable": configurable}

    async def ainvoke(
        self,
        compiled: Any,
        input_state: dict,
        ctx: RunContext,
        *,
        enable_rag: bool = True,
        chat_cfg: Any = None,
    ) -> dict:
        config = self.build_config(
            ctx, enable_rag=enable_rag, chat_cfg=chat_cfg
        )
        return await compiled.ainvoke(input_state, config)

    async def astream(
        self,
        compiled: Any,
        input_state: dict,
        ctx: RunContext,
        *,
        enable_rag: bool = True,
        chat_cfg: Any = None,
    ) -> AsyncIterator[Any]:
        config = self.build_config(
            ctx, enable_rag=enable_rag, chat_cfg=chat_cfg
        )
        async for event in compiled.astream(input_state, config):
            yield event
