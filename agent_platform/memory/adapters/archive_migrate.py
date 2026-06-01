"""SQLite Session Archive → PostgreSQL 迁移。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _normalize_session_row(row: dict) -> dict:
    return {
        "session_id": row["session_id"],
        "user_id": row["user_id"],
        "tenant_id": row["tenant_id"],
        "channel": row.get("channel"),
        "started_at": row["started_at"],
        "status": row.get("status") or "active",
    }


def _normalize_message_row(row: dict) -> dict:
    redacted = row.get("redacted")
    is_redacted = redacted in (True, 1, "1")
    return {
        "message_id": row["message_id"],
        "session_id": row["session_id"],
        "role": row["role"],
        "content": row.get("content"),
        "ts": row["ts"],
        "token_count": int(row.get("token_count") or 0),
        "redacted": is_redacted,
        "metadata_json": row.get("metadata_json"),
    }


def _normalize_tool_call_row(row: dict) -> dict:
    return {
        "call_id": row["call_id"],
        "session_id": row["session_id"],
        "tool_name": row["tool_name"],
        "args_hash": row.get("args_hash"),
        "result_summary": row.get("result_summary"),
        "status": row.get("status"),
        "latency_ms": row.get("latency_ms"),
        "ts": row["ts"],
    }


async def _fetch_all(sqlite_db: Any, table: str) -> List[dict]:
    rows = await sqlite_db.execute(f"SELECT * FROM {table}")
    return [dict(r) for r in rows]


async def migrate_sqlite_to_postgresql(
    sqlite_db: Any,
    pg_db: Any,
    *,
    dry_run: bool = False,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, int]:
    """
    将 SQLite L2 归档（sessions / messages / tool_calls）迁移到 PostgreSQL。

    使用 upsert，可重复执行；已存在的主键会更新 messages，sessions 冲突则跳过。
    """
    sessions = await _fetch_all(sqlite_db, "sessions")
    messages = await _fetch_all(sqlite_db, "messages")
    tool_calls = await _fetch_all(sqlite_db, "tool_calls")

    if tenant_id:
        sessions = [s for s in sessions if s.get("tenant_id") == tenant_id]
        session_ids = {s["session_id"] for s in sessions}
        messages = [
            m
            for m in messages
            if m.get("session_id") in session_ids
        ]
        tool_calls = [
            t
            for t in tool_calls
            if t.get("session_id") in session_ids
        ]
    if user_id:
        sessions = [s for s in sessions if s.get("user_id") == user_id]
        session_ids = {s["session_id"] for s in sessions}
        messages = [m for m in messages if m.get("session_id") in session_ids]
        tool_calls = [t for t in tool_calls if t.get("session_id") in session_ids]

    stats = {
        "sessions": len(sessions),
        "messages": len(messages),
        "tool_calls": len(tool_calls),
        "sessions_written": 0,
        "messages_written": 0,
        "tool_calls_written": 0,
        "errors": 0,
    }

    if dry_run:
        logger.info("Dry run: would migrate %s", stats)
        return stats

    if hasattr(pg_db, "_init_pool"):
        await pg_db._init_pool()

    for row in sessions:
        try:
            await pg_db.upsert_session(_normalize_session_row(row))
            stats["sessions_written"] += 1
        except Exception as e:
            stats["errors"] += 1
            logger.warning("Session migrate failed %s: %s", row.get("session_id"), e)

    for row in messages:
        try:
            await pg_db.insert_message(_normalize_message_row(row))
            stats["messages_written"] += 1
        except Exception as e:
            stats["errors"] += 1
            logger.warning("Message migrate failed %s: %s", row.get("message_id"), e)

    for row in tool_calls:
        try:
            await pg_db.insert_tool_call(_normalize_tool_call_row(row))
            stats["tool_calls_written"] += 1
        except Exception as e:
            stats["errors"] += 1
            logger.warning("Tool call migrate failed %s: %s", row.get("call_id"), e)

    return stats
