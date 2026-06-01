"""L2 冷归档：过期会话导出到对象存储 + 索引表，在线库删除明细。"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional


class SessionColdArchiveService:
    def __init__(
        self,
        archive_db: Any,
        object_store: Any,
        *,
        prefix: str = "l2/cold",
        compress: bool = True,
    ):
        self._db = archive_db
        self._store = object_store
        self._prefix = prefix.rstrip("/")
        self._compress = compress
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _archive_id(session_id: str) -> str:
        return hashlib.sha256(f"cold:{session_id}".encode()).hexdigest()[:16]

    def _object_key(self, tenant_id: str, user_id: str, session_id: str) -> str:
        ext = ".json.gz" if self._compress else ".json"
        return f"{self._prefix}/{tenant_id}/{user_id}/{session_id}{ext}"

    async def _load_session_bundle(self, session_id: str) -> Optional[dict]:
        session = await self._db.select_one(
            "sessions",
            [
                "session_id",
                "user_id",
                "tenant_id",
                "channel",
                "started_at",
                "ended_at",
                "status",
            ],
            {"session_id": session_id},
        )
        if session is None:
            return None

        messages = await self._db.select_many(
            "messages",
            [
                "message_id",
                "session_id",
                "role",
                "content",
                "ts",
                "token_count",
                "redacted",
                "metadata_json",
            ],
            where={"session_id": session_id},
            order_by="ts ASC",
        )
        tool_calls = await self._db.select_many(
            "tool_calls",
            [
                "call_id",
                "session_id",
                "tool_name",
                "args_hash",
                "result_summary",
                "status",
                "latency_ms",
                "ts",
            ],
            where={"session_id": session_id},
            order_by="ts ASC",
        )
        return {
            "session": session,
            "messages": messages,
            "tool_calls": tool_calls,
        }

    async def archive_session(self, session_id: str) -> dict:
        existing = await self._db.get_cold_archive(session_id)
        if existing:
            return {"session_id": session_id, "skipped": True, "object_key": existing["object_key"]}

        bundle = await self._load_session_bundle(session_id)
        if bundle is None:
            raise ValueError(f"Session not found: {session_id}")

        session = bundle["session"]
        tenant_id = session["tenant_id"]
        user_id = session["user_id"]
        payload = {
            "version": 1,
            "archived_at": datetime.now().isoformat(),
            **bundle,
        }
        raw = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        if self._compress:
            body = gzip.compress(raw)
            content_type = "application/gzip"
        else:
            body = raw
            content_type = "application/json"

        checksum = hashlib.sha256(body).hexdigest()
        object_key = self._object_key(tenant_id, user_id, session_id)
        await asyncio.to_thread(
            self._store.upload, object_key, body, content_type
        )

        index_row = {
            "archive_id": self._archive_id(session_id),
            "session_id": session_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "object_key": object_key,
            "message_count": len(bundle["messages"]),
            "tool_call_count": len(bundle["tool_calls"]),
            "payload_bytes": len(body),
            "checksum": checksum,
            "started_at": session.get("started_at"),
            "archived_at": datetime.now().isoformat(),
        }
        await self._db.insert_cold_archive_index(index_row)
        await self._db.delete_online_session(session_id)

        return {
            "session_id": session_id,
            "object_key": object_key,
            "message_count": index_row["message_count"],
            "payload_bytes": index_row["payload_bytes"],
            "skipped": False,
        }

    async def archive_expired_sessions(self, retention_days: int = 90) -> dict:
        session_ids = await self._db.list_expired_session_ids(retention_days)
        archived = 0
        skipped = 0
        errors = 0
        for session_id in session_ids:
            try:
                result = await self.archive_session(session_id)
                if result.get("skipped"):
                    skipped += 1
                else:
                    archived += 1
            except Exception as e:
                errors += 1
                self._logger.warning("Cold archive failed session=%s: %s", session_id, e)
        return {
            "candidates": len(session_ids),
            "archived": archived,
            "skipped": skipped,
            "errors": errors,
        }

    async def list_cold_archives(
        self,
        tenant_id: str,
        user_id: str,
        limit: int = 20,
    ) -> List[dict]:
        return await self._db.list_cold_archives(tenant_id, user_id, limit=limit)

    async def fetch_cold_session(self, session_id: str) -> Optional[dict]:
        index = await self._db.get_cold_archive(session_id)
        if index is None:
            return None

        body = await asyncio.to_thread(self._store.download, index["object_key"])
        if body is None:
            return None

        object_key = index["object_key"]
        if object_key.endswith(".gz"):
            raw = gzip.decompress(body)
        else:
            raw = body
        payload = json.loads(raw.decode("utf-8"))
        payload["cold_index"] = index
        return payload

    async def delete_cold_archives_for_user(
        self, tenant_id: str, user_id: str
    ) -> int:
        rows = await self._db.list_cold_archives(tenant_id, user_id, limit=10_000)
        deleted = 0
        for row in rows:
            key = row.get("object_key")
            if key:
                try:
                    await asyncio.to_thread(self._store.delete, key)
                except Exception as e:
                    self._logger.warning("Delete cold object failed %s: %s", key, e)
            await self._db.delete_cold_archive_index(row["session_id"])
            deleted += 1
        return deleted
