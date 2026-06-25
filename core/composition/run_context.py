from dataclasses import dataclass, field
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.ports.rag import RAGPort
    from core.ports.memory import MemoryPort
    from core.ports.tools import ToolPort
    from core.ports.model import ModelPort
    from core.ports.index import IndexPort
    from core.ports.knowledge_base import KnowledgeBasePort
    from core.ports.policy import PolicyPort
    from core.ports.privacy import PrivacyPort
    from core.ports.observability import ObservabilityPort
    from core.ports.identity import IdentityPort
    from core.ports.skills import SkillPort
    from core.ports.mcp import MCPPort
    from core.ports.storage import CheckpointerPort


@dataclass
class RunContext:
    request: Any

    # ── 领域能力 Port ──
    rag: Optional["RAGPort"] = None
    index: Optional["IndexPort"] = None
    knowledge_base: Optional["KnowledgeBasePort"] = None
    memory: Optional["MemoryPort"] = None
    tools: Optional["ToolPort"] = None
    skills: Optional["SkillPort"] = None
    mcp: Optional["MCPPort"] = None
    models: Optional["ModelPort"] = None

    # ── L1 横切 Port ──
    policy: Optional["PolicyPort"] = None
    privacy: Optional["PrivacyPort"] = None
    observability: Optional["ObservabilityPort"] = None
    identity: Optional["IdentityPort"] = None

    # ── 运行时辅助 ──
    turn_buffer: Optional[Any] = None
    checkpointer: Optional["CheckpointerPort"] = None

    extra: dict = field(default_factory=dict)

    # ── Backward compat ──

    @property
    def index_service(self) -> Optional["IndexPort"]:
        """Backward compat alias for index."""
        return self.index

    @index_service.setter
    def index_service(self, value: Optional["IndexPort"]) -> None:
        self.index = value

    # ── Convenience accessors ──

    def get_model(self, role: str) -> Any:
        if self.models is None:
            raise RuntimeError("ModelPort not initialized")
        return self.models.get_model(role)

    def require_rag(self) -> "RAGPort":
        if self.rag is None:
            raise RuntimeError("RAGPort not initialized")
        return self.rag

    def require_index(self) -> "IndexPort":
        if self.index is None:
            raise RuntimeError("IndexPort not initialized")
        return self.index

    def require_index_service(self) -> "IndexPort":
        return self.require_index()

    def require_knowledge_base(self) -> "KnowledgeBasePort":
        if self.knowledge_base is None:
            raise RuntimeError("KnowledgeBasePort not initialized")
        return self.knowledge_base

    def require_memory(self) -> "MemoryPort":
        if self.memory is None:
            raise RuntimeError("MemoryPort not initialized")
        return self.memory

    def require_tools(self) -> "ToolPort":
        if self.tools is None:
            raise RuntimeError("ToolPort not initialized")
        return self.tools

    def require_skills(self) -> "SkillPort":
        if self.skills is None:
            raise RuntimeError("SkillPort not initialized")
        return self.skills

    def require_mcp(self) -> "MCPPort":
        if self.mcp is None:
            raise RuntimeError("MCPPort not initialized")
        return self.mcp

    def require_checkpointer(self) -> "CheckpointerPort":
        if self.checkpointer is None:
            raise RuntimeError("CheckpointerPort not initialized")
        return self.checkpointer
