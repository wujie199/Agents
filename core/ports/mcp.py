"""MCP 端口。"""
from typing import Protocol, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum


class MCPTransport(str, Enum):
    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


@dataclass
class MCPServerConfig:
    server_id: str
    transport: MCPTransport
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    acl: List[str] = field(default_factory=list)


@dataclass
class MCPToolInfo:
    name: str
    description: str
    input_schema: dict


@dataclass
class MCPToolResult:
    success: bool
    output: Any = None
    error: Optional[str] = None


class MCPPort(Protocol):
    def list_servers(self) -> List[str]:
        ...
    
    def list_tools(self, server_id: str) -> List[MCPToolInfo]:
        ...
    
    def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict,
        timeout_seconds: Optional[int] = None
    ) -> MCPToolResult:
        ...
    
    def health_check(self, server_id: str) -> bool:
        ...
    
    def connect(self, server_id: str) -> bool:
        ...
    
    def disconnect(self, server_id: str) -> None:
        ...
    
    def get_server_info(self, server_id: str) -> Optional[MCPServerConfig]:
        ...
