"""L4 可插拔外部记忆 Provider 契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from core.domain.context import RequestContext


@dataclass
class MemorySnippet:
    """prefetch 返回的单条记忆片段。"""

    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MemoryProvider(Protocol):
    """外部/内置记忆 Provider 契约。"""

    @property
    def name(self) -> str:
        ...

    def is_available(self) -> bool:
        ...

    async def initialize(self, context: RequestContext) -> None:
        ...

    async def shutdown(self) -> None:
        ...

    async def prefetch_turn(
        self, user_message: str, context: RequestContext
    ) -> list[MemorySnippet]:
        ...

    async def sync_turn(
        self,
        user_message: str,
        assistant_message: str,
        messages: list[dict[str, Any]],
        context: RequestContext,
    ) -> None:
        ...

    async def search(
        self, query: str, context: RequestContext, *, limit: int = 5
    ) -> list[MemorySnippet]:
        ...

    async def add(
        self, content: str, context: RequestContext, *, metadata: Optional[dict] = None
    ) -> bool:
        ...

    async def update(
        self, key: str, content: str, context: RequestContext
    ) -> bool:
        ...

    async def delete(self, key: str, context: RequestContext) -> bool:
        ...

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        ...

    async def on_turn_start(
        self, user_message: str, context: RequestContext
    ) -> None:
        ...

    async def on_session_end(self, context: RequestContext) -> None:
        ...

    async def on_session_switch(
        self, old_context: RequestContext, new_context: RequestContext
    ) -> None:
        ...

    async def on_pre_compress(
        self, messages: list[dict[str, Any]], context: RequestContext
    ) -> str:
        ...

    async def on_memory_write(
        self,
        tenant_id: str,
        user_id: str,
        action: str,
        payload: dict[str, Any],
    ) -> None:
        ...

    async def on_delegation(
        self, delegate_to: str, context: RequestContext, *, reason: str = ""
    ) -> None:
        ...


class BaseMemoryProvider:
    """Provider 默认 no-op 实现，子类按需 override。"""

    @property
    def name(self) -> str:
        return "base"

    def is_available(self) -> bool:
        return True

    async def initialize(self, context: RequestContext) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def prefetch_turn(
        self, user_message: str, context: RequestContext
    ) -> list[MemorySnippet]:
        return []

    async def sync_turn(
        self,
        user_message: str,
        assistant_message: str,
        messages: list[dict[str, Any]],
        context: RequestContext,
    ) -> None:
        return None

    async def search(
        self, query: str, context: RequestContext, *, limit: int = 5
    ) -> list[MemorySnippet]:
        return []

    async def add(
        self, content: str, context: RequestContext, *, metadata: Optional[dict] = None
    ) -> bool:
        return False

    async def update(
        self, key: str, content: str, context: RequestContext
    ) -> bool:
        return False

    async def delete(self, key: str, context: RequestContext) -> bool:
        return False

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return []

    async def on_turn_start(
        self, user_message: str, context: RequestContext
    ) -> None:
        return None

    async def on_session_end(self, context: RequestContext) -> None:
        return None

    async def on_session_switch(
        self, old_context: RequestContext, new_context: RequestContext
    ) -> None:
        return None

    async def on_pre_compress(
        self, messages: list[dict[str, Any]], context: RequestContext
    ) -> str:
        return ""

    async def on_memory_write(
        self,
        tenant_id: str,
        user_id: str,
        action: str,
        payload: dict[str, Any],
    ) -> None:
        return None

    async def on_delegation(
        self, delegate_to: str, context: RequestContext, *, reason: str = ""
    ) -> None:
        return None
