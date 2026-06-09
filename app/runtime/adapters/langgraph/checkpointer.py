# -*- coding: utf-8 -*-
"""LangGraph Checkpointer：MemorySaver / SQLite / PostgreSQL（与 L2 同库）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from agent_platform.memory.adapters.config_loader import load_memory_config
from agent_platform.memory.adapters.pg_conn import (
    build_postgres_conn_string,
    is_postgresql_archive,
)

if TYPE_CHECKING:
    from core.composition.run_context import RunContext

logger = logging.getLogger(__name__)

_CHECKPOINTER: Any = None
_PG_SAVER: Any = None
_PG_CM: Any = None


def get_chat_checkpointer(
    *,
    sqlite_path: Optional[str | Path] = None,
) -> Any:
    """返回 LangGraph BaseCheckpointSaver。默认 MemorySaver。"""
    global _CHECKPOINTER
    if sqlite_path is not None:
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            path = Path(sqlite_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            return AsyncSqliteSaver.from_conn_string(str(path))
        except ImportError:
            from langgraph.checkpoint.memory import MemorySaver

            logger.warning(
                "langgraph-checkpoint-sqlite 未安装，回退 MemorySaver"
            )
            return MemorySaver()

    if _CHECKPOINTER is None:
        from langgraph.checkpoint.memory import MemorySaver

        _CHECKPOINTER = MemorySaver()
    return _CHECKPOINTER


async def setup_postgres_checkpointer(conn_string: str) -> Any:
    """初始化 PostgreSQL Checkpointer（与 L2 共用 DSN）。"""
    global _PG_SAVER, _PG_CM
    if _PG_SAVER is not None:
        return _PG_SAVER
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    _PG_CM = AsyncPostgresSaver.from_conn_string(conn_string)
    _PG_SAVER = await _PG_CM.__aenter__()
    await _PG_SAVER.setup()
    logger.info("LangGraph AsyncPostgresSaver 已连接（与 L2 同库）")
    return _PG_SAVER


async def teardown_postgres_checkpointer() -> None:
    global _PG_SAVER, _PG_CM
    if _PG_CM is not None:
        await _PG_CM.__aexit__(None, None, None)
    _PG_SAVER = None
    _PG_CM = None


def _memory_cfg_from_ctx(run_ctx: "RunContext") -> dict:
    extra = run_ctx.extra or {}
    summary = extra.get("memory_config_summary") or {}
    if summary.get("archive_backend"):
        cfg = load_memory_config()
        cfg["archive_backend"] = summary["archive_backend"]
        return cfg
    return load_memory_config()


async def resolve_chat_checkpointer_async(
    run_ctx: Optional["RunContext"] = None,
) -> Any:
    """企业 PG：AsyncPostgresSaver 与 L2 archive 同 DSN。"""
    if run_ctx is None:
        return get_chat_checkpointer()

    cfg = _memory_cfg_from_ctx(run_ctx)
    if is_postgresql_archive(cfg):
        conn = build_postgres_conn_string(cfg)
        if conn:
            try:
                return await setup_postgres_checkpointer(conn)
            except Exception as exc:
                logger.warning(
                    "PostgreSQL checkpointer 初始化失败，回退 SQLite/Memory: %s",
                    exc,
                )

    return resolve_chat_checkpointer(run_ctx)


def resolve_chat_checkpointer(run_ctx: Optional["RunContext"] = None) -> Any:
    """
    同步解析：PG 已初始化则返回 _PG_SAVER；否则 SQLite 文件或 MemorySaver。
    """
    if _PG_SAVER is not None:
        return _PG_SAVER

    if run_ctx is None:
        return get_chat_checkpointer()

    extra = run_ctx.extra or {}
    explicit = extra.get("langgraph_checkpoint_path")
    if explicit:
        return get_chat_checkpointer(sqlite_path=str(explicit))

    relational = extra.get("relational")
    if relational is not None and hasattr(relational, "_db_path"):
        archive_path = Path(relational._db_path)
        cp_path = archive_path.parent / "langgraph_checkpoints.db"
        return get_chat_checkpointer(sqlite_path=str(cp_path))

    data_dir = extra.get("data_dir")
    if data_dir:
        cp_path = Path(data_dir) / "langgraph_checkpoints.db"
        return get_chat_checkpointer(sqlite_path=str(cp_path))

    return get_chat_checkpointer()
