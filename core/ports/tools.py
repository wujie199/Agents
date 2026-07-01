"""工具端口。"""
from typing import Protocol, Any, Optional
from core.domain.context import RequestContext


class ToolPort(Protocol):
    async def invoke(
        self,
        tool_name: str,
        args: dict,
        context: RequestContext
    ) -> Any:
        ...

    async def invoke_batch(
        self,
        tool_name: str,
        args_list: list[dict],
        context: RequestContext
    ) -> list[Any]:
        ...

    def list_tools(self) -> list[str]:
        ...

    def get_schema(self, tool_name: str) -> Optional[dict]:
        ...

    def validate_args(
        self,
        tool_name: str,
        args: dict
    ) -> bool:
        ...
