"""Memory 子系统装配辅助。"""

from __future__ import annotations

from typing import Any, Optional

from agent_platform.memory.adapters.config_loader import load_memory_config
from agent_platform.memory.adapters.hot_memory_file_adapter import HotMemoryFileAdapter
from agent_platform.memory.adapters.relational_checkpointer_adapter import (
    RelationalCheckpointerAdapter,
)
from agent_platform.memory.adapters.turn_buffer import TurnBuffer


def build_turn_buffer(memory: Any, cfg: dict) -> Optional[TurnBuffer]:
    size = int(cfg.get("turn_buffer_flush_size", 0) or 0)
    if size <= 0:
        return None
    return TurnBuffer(memory, flush_size=size)


def build_checkpointer(archive_db: Any) -> Optional[RelationalCheckpointerAdapter]:
    if archive_db is None:
        return None
    return RelationalCheckpointerAdapter(archive_db)


def build_hot_memory(
    cfg: dict[str, Any],
    *,
    archive_db: Any = None,
    store_dir_override: Optional[str] = None,
    database_url: Optional[str] = None,
) -> Any:
    """L1 热记忆：file（默认）、relational（与 L2 同库）或 langgraph Store。"""
    backend = str(cfg.get("l1_store_backend", "file")).lower()
    max_hot = int(cfg.get("hot_memory_max_chars", 2200))
    max_user = int(cfg.get("user_memory_max_chars", 1375))
    use_lock = bool(cfg.get("l1_use_file_lock", True))

    if backend == "langgraph":
        from agent_platform.memory.adapters.hot_memory_langgraph_store_adapter import (
            HotMemoryLangGraphStoreAdapter,
            build_langgraph_memory_store,
        )

        store = build_langgraph_memory_store(cfg, database_url=database_url)
        return HotMemoryLangGraphStoreAdapter(
            store,
            hot_memory_max_chars=max_hot,
            user_memory_max_chars=max_user,
        )

    if backend == "relational" and archive_db is not None:
        from agent_platform.memory.adapters.hot_memory_relational_adapter import (
            HotMemoryRelationalAdapter,
        )

        return HotMemoryRelationalAdapter(
            archive_db=archive_db,
            hot_memory_max_chars=max_hot,
            user_memory_max_chars=max_user,
        )

    store_dir = store_dir_override or cfg.get("store_dir") or "data/memory_dev"
    return HotMemoryFileAdapter(
        store_dir=str(store_dir),
        hot_memory_max_chars=max_hot,
        user_memory_max_chars=max_user,
        use_file_lock=use_lock,
    )


def memory_config_summary_dict(cfg: Optional[dict] = None) -> dict[str, Any]:
    mem = cfg or load_memory_config()
    return {
        "config_path": mem.get("_config_path"),
        "archive_backend": mem.get("archive_backend", "sqlite"),
        "l1_store_backend": mem.get("l1_store_backend", "file"),
        "store_dir": mem.get("store_dir"),
        "enable_cold_archive": bool(mem.get("enable_cold_archive")),
        "enable_session_vector_index": bool(mem.get("enable_session_vector_index")),
    }
