"""向后兼容重导出文件。

所有符号已移至 core.ports.memory 子包，本文件保留以兼容旧 import 路径。
"""

# 重导出所有 DTO 和 MemoryPort，保持旧路径可用
from core.ports.memory.dtos import (  # noqa: F401
    PromptMemorySnapshot,
    TurnRecord,
    ToolCallRecord,
    MemoryDelta,
    SkillSummary,
    SkillOutcome,
    SessionFragment,
    SessionSearchResult,
)
from core.ports.memory import MemoryPort  # noqa: F401
from core.ports.memory.hot import HotMemoryPort, HotMemoryStore, HotMemoryCompressor  # noqa: F401
from core.ports.memory.archive import ArchivePort  # noqa: F401
from core.ports.memory.skill import SkillMemoryPort  # noqa: F401
from core.ports.memory.external import ExternalProfilePort, ExternalMemoryProvider, Entity, Fact  # noqa: F401
from core.ports.memory.search import SessionSearchPort  # noqa: F401
from core.ports.memory.admin import MemoryAdminPort  # noqa: F401
from core.ports.memory.summarizer import MemorySummarizer  # noqa: F401
