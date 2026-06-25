"""Memory Port 聚合 — Hermes 四层记忆 + 搜索 + 运维。

通过 Protocol 多继承组合所有子端口，保持向后兼容：
    from core.ports.memory import MemoryPort
"""

from typing import Protocol

from core.ports.memory.dtos import (  # noqa: F401 — 重导出所有 DTO
    PromptMemorySnapshot,
    TurnRecord,
    ToolCallRecord,
    MemoryDelta,
    SkillSummary,
    SkillOutcome,
    SessionFragment,
    SessionSearchResult,
)
from core.ports.memory.hot import HotMemoryPort, HotMemoryStore, HotMemoryCompressor  # noqa: F401
from core.ports.memory.archive import ArchivePort  # noqa: F401
from core.ports.memory.skill import SkillMemoryPort  # noqa: F401
from core.ports.memory.external import ExternalProfilePort, ExternalMemoryProvider, Entity, Fact  # noqa: F401
from core.ports.memory.search import SessionSearchPort  # noqa: F401
from core.ports.memory.admin import MemoryAdminPort  # noqa: F401
from core.ports.memory.summarizer import MemorySummarizer  # noqa: F401


class MemoryPort(
    HotMemoryPort,
    ArchivePort,
    SkillMemoryPort,
    ExternalProfilePort,
    SessionSearchPort,
    MemoryAdminPort,
    Protocol,
):
    """聚合记忆端口 — 完整的 Hermes 四层记忆 + 搜索 + 运维。

    子协议：
        HotMemoryPort      — L1 热记忆（提示快照 + 增量写入）
        ArchivePort        — L2 冷档案（会话 + 轮次持久化）
        SkillMemoryPort    — L3 技能记忆（搜索/执行/发布）
        ExternalProfilePort — L4 外部画像（实体解析 + Profile）
        SessionSearchPort  — 会话搜索（全文检索 + 摘要）
        MemoryAdminPort    — 运维管理（清理/索引/归档）
    """

    pass


__all__ = [
    # 聚合端口
    "MemoryPort",
    # 子端口
    "HotMemoryPort",
    "HotMemoryStore",
    "HotMemoryCompressor",
    "ArchivePort",
    "SkillMemoryPort",
    "ExternalProfilePort",
    "ExternalMemoryProvider",
    "SessionSearchPort",
    "MemoryAdminPort",
    "MemorySummarizer",
    # DTO
    "PromptMemorySnapshot",
    "TurnRecord",
    "ToolCallRecord",
    "MemoryDelta",
    "SkillSummary",
    "SkillOutcome",
    "SessionFragment",
    "SessionSearchResult",
    "Entity",
    "Fact",
]
