"""基于 RelationalPort 的 Checkpointer 实现。"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, List, Optional


class RelationalCheckpointerAdapter:
    TABLE = "graph_checkpoints"

    def __init__(self, archive_db: Any):
        self._db = archive_db
        self._logger = logging.getLogger(__name__)

    async def save(
        self,
        thread_id: str,
        tenant_id: str,
        state: dict,
        *,
        session_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        checkpoint_ns: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        checkpoint_id = str(uuid.uuid4())
        row = {
            "checkpoint_id": checkpoint_id,
            "thread_id": thread_id,
            "tenant_id": tenant_id,
            "session_id": session_id or thread_id,
            "checkpoint_ns": checkpoint_ns or "",
            "parent_id": parent_id,
            "state_json": json.dumps(state, ensure_ascii=False, default=str),
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
            "created_at": datetime.now().isoformat(),
        }
        await self._db.insert(self.TABLE, row)
        return checkpoint_id

    async def load(
        self,
        thread_id: str,
        tenant_id: str,
        *,
        checkpoint_ns: str = "",
    ) -> Optional[dict]:
        rows = await self._db.select_many(
            self.TABLE,
            [
                "checkpoint_id",
                "state_json",
                "metadata_json",
                "created_at",
                "parent_id",
            ],
            where={
                "thread_id": thread_id,
                "tenant_id": tenant_id,
                "checkpoint_ns": checkpoint_ns or "",
            },
            order_by="created_at DESC",
            limit=1,
        )
        if not rows:
            return None
        row = rows[0]
        try:
            state = json.loads(row["state_json"])
        except Exception as e:
            self._logger.warning("Checkpoint state parse failed: %s", e)
            return None
        return {
            "checkpoint_id": row["checkpoint_id"],
            "state": state,
            "metadata": json.loads(row.get("metadata_json") or "{}"),
            "created_at": row.get("created_at"),
            "parent_id": row.get("parent_id"),
        }

    async def list_threads(
        self, tenant_id: str, *, limit: int = 20
    ) -> List[dict]:
        if hasattr(self._db, "execute"):
            query = f"""
                SELECT thread_id, session_id, MAX(created_at) AS last_checkpoint_at,
                       COUNT(*) AS checkpoint_count
                FROM {self.TABLE}
                WHERE tenant_id = :tenant_id
                GROUP BY thread_id, session_id
                ORDER BY last_checkpoint_at DESC
                LIMIT :limit
            """
            rows = await self._db.execute(
                query, {"tenant_id": tenant_id, "limit": limit}
            )
            if isinstance(rows, list):
                return [dict(r) for r in rows]
        return []

    async def health_check(self) -> dict:
        try:
            count = 0
            if hasattr(self._db, "execute"):
                rows = await self._db.execute(
                    f"SELECT COUNT(*) AS c FROM {self.TABLE}",
                    {},
                )
                if rows:
                    row = rows[0]
                    if isinstance(row, dict):
                        count = int(row.get("c", 0))
                    else:
                        count = int(row[0])
            return {
                "status": "healthy",
                "table": self.TABLE,
                "checkpoint_count": count,
            }
        except Exception as e:
            return {"status": "unhealthy", "table": self.TABLE, "error": str(e)}

    async def purge_thread(
        self, thread_id: str, tenant_id: str, *, checkpoint_ns: str = ""
    ) -> int:
        if not hasattr(self._db, "delete"):
            return 0
        return await self._db.delete(
            self.TABLE,
            {
                "thread_id": thread_id,
                "tenant_id": tenant_id,
                "checkpoint_ns": checkpoint_ns or "",
            },
        )

    async def purge_older_than(self, retention_days: int) -> int:
        """删除早于 retention_days 的 checkpoint 行。"""
        days = max(1, int(retention_days))
        if not hasattr(self._db, "_get_connection"):
            return 0
        backend = type(self._db).__name__.lower()
        try:
            async with self._db._get_connection() as conn:
                if "postgres" in backend:
                    status = await conn.execute(
                        f"""
                        DELETE FROM {self.TABLE}
                        WHERE created_at::timestamptz < NOW() - ($1::text || ' days')::interval
                        """,
                        str(days),
                    )
                    if status and isinstance(status, str) and status.startswith("DELETE"):
                        return int(status.split()[-1])
                    return 0
                cursor = await conn.cursor()
                await cursor.execute(
                    f"""
                    DELETE FROM {self.TABLE}
                    WHERE created_at < datetime('now', '-' || ? || ' days')
                    """,
                    (days,),
                )
                return int(cursor.rowcount or 0)
        except Exception as e:
            self._logger.warning("Checkpoint purge failed: %s", e)
            return 0
