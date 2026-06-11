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
        search_scan_limit: int = 100,
        encrypt_at_rest: bool = False,
        encryption_key: Optional[str] = None,
    ):
        self._db = archive_db
        self._store = object_store
        self._prefix = prefix.rstrip("/")
        self._compress = compress
        self._search_scan_limit = max(1, search_scan_limit)
        self._encrypt_at_rest = encrypt_at_rest
        self._encryption_key = encryption_key
        self._logger = logging.getLogger(__name__)

    def health(self) -> dict:
        if self._store is None:
            return {"status": "not_configured"}
        if hasattr(self._store, "health"):
            raw = self._store.health()
            if isinstance(raw, dict):
                return raw
        return {"status": "unknown", "type": type(self._store).__name__}

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

    async def _object_exists(self, object_key: str) -> bool:
        body = await asyncio.to_thread(self._store.download, object_key)
        return body is not None

    async def archive_session(self, session_id: str) -> dict:
        existing = await self._db.get_cold_archive(session_id)
        if existing:
            object_key = existing["object_key"]
            if await self._object_exists(object_key):
                return {
                    "session_id": session_id,
                    "skipped": True,
                    "object_key": object_key,
                }
            self._logger.warning(
                "Cold index exists but object missing, re-archiving session=%s",
                session_id,
            )
            await self._db.delete_cold_archive_index(session_id)

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

        object_key = self._object_key(tenant_id, user_id, session_id)
        if self._encrypt_at_rest and self._encryption_key:
            from agent_platform.memory.adapters.field_crypto import encrypt_bytes

            body = encrypt_bytes(body, self._encryption_key)
        checksum = hashlib.sha256(body).hexdigest()
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
        try:
            await self._db.insert_cold_archive_index(index_row)
            await self._index_cold_search_rows(bundle, tenant_id, user_id, session_id)
            await self._db.delete_online_session(session_id)
        except Exception:
            try:
                await asyncio.to_thread(self._store.delete, object_key)
            except Exception as cleanup_err:
                self._logger.warning(
                    "Rollback cold object failed session=%s: %s",
                    session_id,
                    cleanup_err,
                )
            raise

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

        expected_checksum = index.get("checksum")
        if expected_checksum:
            actual = hashlib.sha256(body).hexdigest()
            if actual != expected_checksum:
                self._logger.error(
                    "Cold archive checksum mismatch session=%s expected=%s actual=%s",
                    session_id,
                    expected_checksum,
                    actual,
                )
                return None

        if self._encrypt_at_rest and self._encryption_key:
            from agent_platform.memory.adapters.field_crypto import decrypt_bytes

            body = decrypt_bytes(body, self._encryption_key)

        object_key = index["object_key"]
        if object_key.endswith(".gz"):
            raw = gzip.decompress(body)
        else:
            raw = body
        payload = json.loads(raw.decode("utf-8"))
        payload["cold_index"] = index
        return payload

    @staticmethod
    def build_search_rows(
        bundle: dict,
        tenant_id: str,
        user_id: str,
        session_id: str,
    ) -> List[dict]:
        rows: List[dict] = []
        for msg in bundle.get("messages") or []:
            if msg.get("redacted"):
                continue
            content = (msg.get("content") or "").strip()
            if not content or content == "[redacted]":
                continue
            rows.append(
                {
                    "record_id": str(msg.get("message_id") or ""),
                    "session_id": session_id,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "role": str(msg.get("role") or "user"),
                    "content": content,
                    "ts": msg.get("ts"),
                    "record_type": "message",
                }
            )

        for tc in bundle.get("tool_calls") or []:
            summary = (tc.get("result_summary") or "").strip()
            tool_name = tc.get("tool_name") or ""
            if summary == "[redacted]":
                continue
            content = f"{tool_name}: {summary}".strip(": ")
            if not content:
                continue
            rows.append(
                {
                    "record_id": str(tc.get("call_id") or ""),
                    "session_id": session_id,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "role": "tool",
                    "content": content,
                    "ts": tc.get("ts"),
                    "record_type": "tool_call",
                }
            )
        return [r for r in rows if r["record_id"]]

    async def _index_cold_search_rows(
        self,
        bundle: dict,
        tenant_id: str,
        user_id: str,
        session_id: str,
    ) -> None:
        inserter = getattr(self._db, "insert_cold_archive_search_rows", None)
        if inserter is None:
            return
        rows = self.build_search_rows(bundle, tenant_id, user_id, session_id)
        if rows:
            await inserter(rows)

    async def _search_cold_via_db(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        *,
        session_id: Optional[str],
        limit: int,
    ) -> List[dict]:
        searcher = getattr(self._db, "search_cold_archive_messages", None)
        if searcher is None:
            return []
        try:
            return await searcher(
                tenant_id,
                user_id,
                query,
                session_id=session_id,
                limit=limit,
            )
        except Exception as e:
            self._logger.warning("Cold archive DB search failed: %s", e)
            return []

    @staticmethod
    def _score_text(text: str, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        lower = text.lower()
        return sum(1.0 for t in tokens if t in lower) / len(tokens)

    async def _rows_for_cold_search(
        self,
        tenant_id: str,
        user_id: str,
        session_id: Optional[str],
        scan_limit: int,
    ) -> List[dict]:
        if session_id:
            row = await self._db.get_cold_archive(session_id)
            if row is None:
                return []
            if row.get("tenant_id") != tenant_id or row.get("user_id") != user_id:
                return []
            return [row]
        return await self._db.list_cold_archives(
            tenant_id, user_id, limit=scan_limit
        )

    def _scan_hits_from_payload(
        self,
        payload: dict,
        sid: str,
        tokens: List[str],
    ) -> List[dict]:
        hits: List[dict] = []
        for msg in payload.get("messages") or []:
            if msg.get("redacted"):
                continue
            content = (msg.get("content") or "").strip()
            if not content or content == "[redacted]":
                continue
            score = self._score_text(content, tokens)
            if score <= 0:
                continue
            hits.append(
                {
                    **msg,
                    "session_id": sid,
                    "score": score,
                    "source": "cold",
                }
            )

        for tc in payload.get("tool_calls") or []:
            summary = (tc.get("result_summary") or "").strip()
            tool_name = tc.get("tool_name") or ""
            text = f"{tool_name} {summary}"
            score = self._score_text(text, tokens)
            if score <= 0:
                continue
            hits.append(
                {
                    "message_id": tc.get("call_id"),
                    "session_id": sid,
                    "role": "tool",
                    "content": f"{tool_name}: {summary}".strip(": "),
                    "ts": tc.get("ts"),
                    "score": score,
                    "source": "cold",
                }
            )
        return hits

    async def _search_cold_via_blob_scan(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        *,
        session_id: Optional[str],
        limit: int,
        scan_limit: int,
    ) -> List[dict]:
        """Legacy fallback：逐对象下载扫描（无 DB 索引的旧归档）。"""
        tokens = [t for t in query.lower().split() if t]
        if not tokens:
            return []

        rows = await self._rows_for_cold_search(
            tenant_id, user_id, session_id, scan_limit
        )
        hits: List[dict] = []
        for row in rows:
            sid = row.get("session_id")
            if not sid:
                continue
            payload = await self.fetch_cold_session(sid)
            if not payload:
                continue
            hits.extend(self._scan_hits_from_payload(payload, sid, tokens))

        hits.sort(key=lambda h: h.get("score", 0), reverse=True)
        return hits[:limit]

    async def search_cold_archives(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        *,
        session_id: Optional[str] = None,
        limit: int = 10,
        scan_limit: Optional[int] = None,
    ) -> List[dict]:
        """冷归档检索：优先 DB FTS 索引，无结果时 fallback 对象扫描。"""
        if not (query or "").strip():
            return []

        db_hits = await self._search_cold_via_db(
            query, tenant_id, user_id, session_id=session_id, limit=limit
        )
        if db_hits:
            return db_hits

        effective_scan = scan_limit if scan_limit is not None else self._search_scan_limit
        return await self._search_cold_via_blob_scan(
            query,
            tenant_id,
            user_id,
            session_id=session_id,
            limit=limit,
            scan_limit=effective_scan,
        )

    async def backfill_search_index(
        self,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """为历史冷归档补建 DB 检索索引。"""
        lister = getattr(self._db, "list_cold_archive_sessions", None)
        if lister is None:
            return {"indexed": 0, "skipped": 0, "errors": 0, "reason": "not_supported"}

        indexed = 0
        skipped = 0
        errors = 0
        offset = 0
        batch_size = max(1, limit)

        while True:
            if session_id:
                row = await self._db.get_cold_archive(session_id)
                rows = [row] if row else []
            else:
                rows = await lister(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    limit=batch_size,
                    offset=offset,
                )
            if not rows:
                break

            for row in rows:
                sid = row.get("session_id")
                if not sid:
                    continue
                if tenant_id and row.get("tenant_id") != tenant_id:
                    continue
                if user_id and row.get("user_id") != user_id:
                    continue

                counter = getattr(self._db, "count_cold_archive_search_rows", None)
                if counter and not force:
                    existing = await counter(sid)
                    if existing > 0:
                        skipped += 1
                        continue

                if dry_run:
                    indexed += 1
                    continue

                try:
                    if force:
                        deleter = getattr(self._db, "delete_cold_archive_search", None)
                        if deleter:
                            await deleter(sid)
                    payload = await self.fetch_cold_session(sid)
                    if not payload:
                        errors += 1
                        continue
                    bundle = {
                        "messages": payload.get("messages") or [],
                        "tool_calls": payload.get("tool_calls") or [],
                    }
                    await self._index_cold_search_rows(
                        bundle,
                        row["tenant_id"],
                        row["user_id"],
                        sid,
                    )
                    indexed += 1
                except Exception as e:
                    errors += 1
                    self._logger.warning(
                        "Cold search backfill failed session=%s: %s", sid, e
                    )

            if session_id:
                break
            offset += len(rows)
            if len(rows) < batch_size:
                break

        return {
            "indexed": indexed,
            "skipped": skipped,
            "errors": errors,
            "dry_run": dry_run,
        }

    async def delete_cold_archives_for_user(
        self, tenant_id: str, user_id: str
    ) -> int:
        from agent_platform.memory.adapters.compliance_utils import append_audit_log

        rows = await self._db.list_cold_archives(tenant_id, user_id, limit=10_000)
        deleted = 0
        for row in rows:
            checksum = row.get("checksum")
            if checksum:
                await append_audit_log(
                    self._db,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    resource_type="cold_archive",
                    resource_id=str(row.get("session_id") or ""),
                    content_hash=str(checksum),
                    action="delete",
                    meta={"object_key": row.get("object_key")},
                )
            key = row.get("object_key")
            if key:
                try:
                    await asyncio.to_thread(self._store.delete, key)
                except Exception as e:
                    self._logger.warning("Delete cold object failed %s: %s", key, e)
                    continue
            await self._db.delete_cold_archive_index(row["session_id"])
            deleted += 1
        return deleted
