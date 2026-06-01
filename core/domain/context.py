from dataclasses import dataclass, field
from typing import Optional, Set, List, Any


@dataclass(frozen=True)
class ACL:
    doc_ids: frozenset[str] = field(default_factory=frozenset)
    tool_names: frozenset[str] = field(default_factory=frozenset)
    mcp_servers: frozenset[str] = field(default_factory=frozenset)

    def can_access_doc(self, doc_id: str) -> bool:
        return doc_id in self.doc_ids

    def can_use_tool(self, tool_name: str) -> bool:
        return tool_name in self.tool_names

    def can_use_mcp(self, server_id: str) -> bool:
        return server_id in self.mcp_servers


@dataclass(frozen=True)
class RequestContext:
    tenant_id: str
    user_id: str
    session_id: str
    trace_id: str
    channel: str
    acl: ACL = field(default_factory=ACL)

    def __post_init__(self):
        if not self.tenant_id:
            raise ValueError("tenant_id is required")
        if not self.user_id:
            raise ValueError("user_id is required")
        if not self.session_id:
            raise ValueError("session_id is required")
        if not self.trace_id:
            raise ValueError("trace_id is required")
