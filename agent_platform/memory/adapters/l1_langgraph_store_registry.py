# -*- coding: utf-8
"""LangGraph PostgresStore 生命周期（与 Checkpointer 模式一致）。"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PG_STORE: Any = None
_PG_STORE_CM: Any = None


def get_postgres_memory_store() -> Optional[Any]:
    return _PG_STORE


def setup_postgres_memory_store(conn_string: str) -> Any:
    """初始化 PostgresStore 并建表。"""
    global _PG_STORE, _PG_STORE_CM
    if _PG_STORE is not None:
        return _PG_STORE
    from langgraph.store.postgres import PostgresStore

    _PG_STORE_CM = PostgresStore.from_conn_string(conn_string)
    _PG_STORE = _PG_STORE_CM.__enter__()
    _PG_STORE.setup()
    logger.info("LangGraph PostgresStore 已连接（L1 langgraph 后端）")
    return _PG_STORE


def teardown_postgres_memory_store() -> None:
    global _PG_STORE, _PG_STORE_CM
    if _PG_STORE_CM is not None:
        try:
            _PG_STORE_CM.__exit__(None, None, None)
        except Exception as exc:
            logger.debug("PostgresStore teardown: %s", exc)
    _PG_STORE = None
    _PG_STORE_CM = None
