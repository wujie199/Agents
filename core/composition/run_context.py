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


@dataclass
class RunContext:
    request: Any

    rag: Optional["RAGPort"] = None
    index: Optional["IndexPort"] = None
    knowledge_base: Optional["KnowledgeBasePort"] = None
    memory: Optional["MemoryPort"] = None
    tools: Optional["ToolPort"] = None
    skills: Optional[Any] = None
    mcp: Optional[Any] = None
    models: Optional["ModelPort"] = None

    policy: Optional["PolicyPort"] = None
    privacy: Optional["PrivacyPort"] = None
    observability: Optional["ObservabilityPort"] = None
    identity: Optional["IdentityPort"] = None

    turn_buffer: Optional[Any] = None
    checkpointer: Optional[Any] = None

    extra: dict = field(default_factory=dict)

    @property
    def index_service(self) -> Optional["IndexPort"]:
        """Backward compat alias for index."""
        return self.index

    @index_service.setter
    def index_service(self, value: Optional["IndexPort"]) -> None:
        self.index = value

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
