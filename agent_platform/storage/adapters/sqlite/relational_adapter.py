import asyncio
import aiosqlite
import re
from typing import Optional, List, Any
from contextlib import asynccontextmanager
from pathlib import Path
import time
import logging


class AsyncSQLiteRelationalAdapter:
    def __init__(
        self,
        db_path: str = "data/session_archive.db",
        pool_size: int = 5,
        timeout: float = 30.0
    ):
        self._db_path = db_path
        self._pool_size = pool_size
        self._timeout = timeout
        self._pool: List[aiosqlite.Connection] = []
        self._semaphore = asyncio.Semaphore(pool_size)
        self._logger = logging.getLogger(__name__)
        
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    async def _init_pool(self) -> None:
        if self._pool:
            return
        
        for _ in range(self._pool_size):
            conn = await aiosqlite.connect(self._db_path)
            conn.row_factory = aiosqlite.Row
            await self._init_tables(conn)
            self._pool.append(conn)
        
        self._logger.info(f"Connection pool initialized with {self._pool_size} connections")
    
    async def _init_tables(self, conn: aiosqlite.Connection) -> None:
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                channel TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT DEFAULT 'active'
            );
            
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                ts TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                redacted INTEGER DEFAULT 0,
                metadata_json TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            
            CREATE TABLE IF NOT EXISTS tool_calls (
                call_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                args_hash TEXT,
                result_summary TEXT,
                status TEXT,
                latency_ms INTEGER,
                ts TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts);
            CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id);

            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                title TEXT,
                content TEXT,
                metadata TEXT,
                updated_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (doc_id, tenant_id)
            );
            CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents(tenant_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content,
                message_id UNINDEXED,
                session_id UNINDEXED,
                tokenize='unicode61'
            );

            CREATE TABLE IF NOT EXISTS cold_archive_sessions (
                archive_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL UNIQUE,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                object_key TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                payload_bytes INTEGER DEFAULT 0,
                checksum TEXT,
                started_at TEXT,
                archived_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cold_archive_tenant_user
                ON cold_archive_sessions(tenant_id, user_id);
            CREATE INDEX IF NOT EXISTS idx_cold_archive_session
                ON cold_archive_sessions(session_id);

            CREATE TABLE IF NOT EXISTS cold_archive_search (
                record_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts TEXT,
                record_type TEXT DEFAULT 'message'
            );
            CREATE INDEX IF NOT EXISTS idx_cas_session
                ON cold_archive_search(session_id);
            CREATE INDEX IF NOT EXISTS idx_cas_tenant_user
                ON cold_archive_search(tenant_id, user_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS cold_archive_fts USING fts5(
                content,
                record_id UNINDEXED,
                session_id UNINDEXED,
                tenant_id UNINDEXED,
                user_id UNINDEXED,
                tokenize='unicode61'
            );

            CREATE TABLE IF NOT EXISTS compliance_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                action TEXT NOT NULL,
                ts TEXT NOT NULL,
                meta_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_audit_tenant_user
                ON compliance_audit_log(tenant_id, user_id);

            CREATE TABLE IF NOT EXISTS graph_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                session_id TEXT,
                checkpoint_ns TEXT DEFAULT '',
                parent_id TEXT,
                state_json TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_gc_thread
                ON graph_checkpoints(thread_id, tenant_id);

            CREATE TABLE IF NOT EXISTS skill_runs (
                run_id TEXT PRIMARY KEY,
                skill_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT,
                trace_id TEXT,
                success INTEGER NOT NULL DEFAULT 0,
                steps_executed INTEGER DEFAULT 0,
                error TEXT,
                inputs_json TEXT,
                outputs_json TEXT,
                ts TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_skill_runs_tenant_user
                ON skill_runs(tenant_id, user_id);
            CREATE INDEX IF NOT EXISTS idx_skill_runs_skill
                ON skill_runs(skill_id);

            CREATE TABLE IF NOT EXISTS hot_memory_docs (
                tenant_id TEXT NOT NULL,
                doc_kind TEXT NOT NULL,
                user_id TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (tenant_id, doc_kind, user_id)
            );
        """)
        await self._migrate_schema(conn)

    async def _migrate_schema(self, conn: aiosqlite.Connection) -> None:
        for stmt in (
            "ALTER TABLE messages ADD COLUMN content_hash TEXT",
            "ALTER TABLE tool_calls ADD COLUMN result_hash TEXT",
        ):
            try:
                await conn.execute(stmt)
            except Exception:
                pass
    
    @asynccontextmanager
    async def _get_connection(self):
        await self._init_pool()
        
        async with self._semaphore:
            if not self._pool:
                conn = await aiosqlite.connect(self._db_path)
                conn.row_factory = aiosqlite.Row
            else:
                conn = self._pool.pop()
            
            try:
                yield conn
                await conn.commit()
            except Exception as e:
                await conn.rollback()
                self._logger.error(f"Transaction failed: {e}")
                raise
            finally:
                self._pool.append(conn)
    
    async def execute(self, query: str, params: Optional[dict] = None) -> Any:
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            if params:
                await cursor.execute(query, params)
            else:
                await cursor.execute(query)
            return await cursor.fetchall()
    
    async def insert(self, table: str, data: dict) -> str:
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(query, data)
            return cursor.lastrowid
    
    async def update(self, table: str, data: dict, where: dict) -> int:
        set_clause = ", ".join(f"{k} = :{k}" for k in data.keys())
        where_clause = " AND ".join(f"{k} = :where_{k}" for k in where.keys())
        query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        
        params = data.copy()
        for k, v in where.items():
            params[f"where_{k}"] = v
        
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(query, params)
            return cursor.rowcount
    
    async def select_one(self, table: str, columns: List[str], where: dict) -> Optional[dict]:
        cols = ", ".join(columns)
        where_clause = " AND ".join(f"{k} = :{k}" for k in where.keys())
        query = f"SELECT {cols} FROM {table} WHERE {where_clause} LIMIT 1"
        
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(query, where)
            row = await cursor.fetchone()
            if row:
                return dict(row)
        return None
    
    async def select_many(
        self,
        table: str,
        columns: List[str],
        where: Optional[dict] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[dict]:
        cols = ", ".join(columns)
        query = f"SELECT {cols} FROM {table}"
        
        params = {}
        if where:
            where_clause = " AND ".join(f"{k} = :{k}" for k in where.keys())
            query += f" WHERE {where_clause}"
            params = where
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        if offset:
            query += f" OFFSET {offset}"
        
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(query, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    async def upsert_session(self, data: dict) -> None:
        query = """
            INSERT INTO sessions (
                session_id, user_id, tenant_id, channel, started_at, status
            ) VALUES (
                :session_id, :user_id, :tenant_id, :channel, :started_at, :status
            )
            ON CONFLICT(session_id) DO NOTHING
        """
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(query, data)

    async def end_session(self, session_id: str, status: str = "closed") -> None:
        from datetime import datetime

        await self.update(
            "sessions",
            {"ended_at": datetime.now().isoformat(), "status": status},
            {"session_id": session_id},
        )

    async def list_sessions(
        self,
        tenant_id: str,
        user_id: str,
        limit: int = 20,
    ) -> List[dict]:
        return await self.select_many(
            "sessions",
            ["session_id", "user_id", "tenant_id", "channel", "started_at", "ended_at", "status"],
            where={"tenant_id": tenant_id, "user_id": user_id},
            order_by="started_at DESC",
            limit=limit,
        )

    async def insert_message(self, data: dict) -> str:
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        query = f"INSERT OR REPLACE INTO messages ({columns}) VALUES ({placeholders})"

        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(query, data)
            await self._sync_message_fts(
                cursor,
                data["message_id"],
                data["session_id"],
                data.get("content") or "",
            )
        return str(data["message_id"])

    async def insert_tool_call(self, data: dict) -> str:
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        query = f"INSERT OR IGNORE INTO tool_calls ({columns}) VALUES ({placeholders})"
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(query, data)
        return str(data["call_id"])

    @staticmethod
    async def _sync_message_fts(
        cursor: aiosqlite.Cursor,
        message_id: str,
        session_id: str,
        content: str,
    ) -> None:
        await cursor.execute(
            "DELETE FROM messages_fts WHERE message_id = :message_id",
            {"message_id": message_id},
        )
        if not content or content == "[redacted]":
            return
        await cursor.execute(
            "INSERT INTO messages_fts (message_id, session_id, content) "
            "VALUES (:message_id, :session_id, :content)",
            {
                "message_id": message_id,
                "session_id": session_id,
                "content": content,
            },
        )

    @staticmethod
    def _build_fts_query(query: str) -> str:
        from agent_platform.memory.adapters.session_search_terms import (
            build_fts_match_query,
        )

        return build_fts_match_query(query)

    async def _search_messages_fts(
        self,
        session_id: Optional[str],
        user_id: Optional[str],
        tenant_id: Optional[str],
        query: str,
        limit: int,
    ) -> List[dict]:
        conditions = ["messages_fts MATCH :fts_query"]
        params: dict = {
            "fts_query": self._build_fts_query(query),
            "limit": limit,
        }

        if session_id:
            conditions.append("m.session_id = :session_id")
            params["session_id"] = session_id
        if user_id:
            conditions.append("s.user_id = :user_id")
            params["user_id"] = user_id
        if tenant_id:
            conditions.append("s.tenant_id = :tenant_id")
            params["tenant_id"] = tenant_id
        conditions.append("(m.redacted IS NULL OR m.redacted = 0)")

        where = " AND ".join(conditions)
        sql = f"""
            SELECT m.* FROM messages m
            JOIN sessions s ON m.session_id = s.session_id
            JOIN messages_fts f ON m.message_id = f.message_id
            WHERE {where}
            ORDER BY m.ts DESC
            LIMIT :limit
        """
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def _search_messages_like(
        self,
        session_id: Optional[str],
        user_id: Optional[str],
        tenant_id: Optional[str],
        query: Optional[str],
        limit: int,
    ) -> List[dict]:
        if user_id or tenant_id:
            sql = """
                SELECT m.* FROM messages m
                JOIN sessions s ON m.session_id = s.session_id
                WHERE 1=1
            """
        else:
            sql = "SELECT * FROM messages WHERE 1=1"

        params: dict = {}

        if session_id:
            sql += " AND m.session_id = :session_id" if "JOIN" in sql else " AND session_id = :session_id"
            params["session_id"] = session_id
        if user_id:
            sql += " AND s.user_id = :user_id"
            params["user_id"] = user_id
        if tenant_id:
            sql += " AND s.tenant_id = :tenant_id"
            params["tenant_id"] = tenant_id
        if "JOIN" in sql:
            sql += " AND (m.redacted IS NULL OR m.redacted = 0)"
        else:
            sql += " AND (redacted IS NULL OR redacted = 0)"
        if query:
            from agent_platform.memory.adapters.session_search_terms import (
                extract_search_terms,
            )

            col = "m.content" if "JOIN" in sql else "content"
            terms = extract_search_terms(query)
            ors = []
            for i, term in enumerate(terms):
                key = f"qt{i}"
                ors.append(f"{col} LIKE :{key}")
                params[key] = f"%{term}%"
            sql += f" AND ({' OR '.join(ors)})"

        sql += " ORDER BY m.ts DESC LIMIT :limit" if "JOIN" in sql else " ORDER BY ts DESC LIMIT :limit"
        params["limit"] = limit

        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def search_messages(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        if query:
            try:
                rows = await self._search_messages_fts(
                    session_id, user_id, tenant_id, query, limit
                )
                if rows:
                    return rows
            except Exception as e:
                self._logger.debug("FTS search fallback to LIKE: %s", e)
        return await self._search_messages_like(
            session_id, user_id, tenant_id, query, limit
        )

    async def search_tool_calls(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        if not query:
            return []
        tokens = [t for t in query.split() if t]
        token_conditions = []
        params: dict = {"limit": limit}
        for i, token in enumerate(tokens):
            key = f"p{i}"
            token_conditions.append(
                f"(t.result_summary LIKE :{key} OR t.tool_name LIKE :{key})"
            )
            params[key] = f"%{token}%"
        conditions = [f"({' AND '.join(token_conditions)})"] if token_conditions else []
        if session_id:
            conditions.append("t.session_id = :session_id")
            params["session_id"] = session_id
        if user_id:
            conditions.append("s.user_id = :user_id")
            params["user_id"] = user_id
        if tenant_id:
            conditions.append("s.tenant_id = :tenant_id")
            params["tenant_id"] = tenant_id

        where = " AND ".join(conditions)
        sql = f"""
            SELECT t.call_id AS message_id, t.session_id, 'tool' AS role,
                   (t.tool_name || ': ' || COALESCE(t.result_summary, '')) AS content,
                   t.ts, s.tenant_id, s.user_id
            FROM tool_calls t
            JOIN sessions s ON t.session_id = s.session_id
            WHERE {where}
            ORDER BY t.ts DESC
            LIMIT :limit
        """
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def list_messages_for_reindex(
        self,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[dict]:
        sql = """
            SELECT m.message_id, m.session_id, m.role, m.content, m.ts,
                   m.redacted, s.tenant_id, s.user_id
            FROM messages m
            JOIN sessions s ON m.session_id = s.session_id
            WHERE (m.redacted IS NULL OR m.redacted = 0)
              AND m.content IS NOT NULL
              AND m.content <> ''
              AND m.content <> '[redacted]'
        """
        params: dict = {}

        if tenant_id:
            sql += " AND s.tenant_id = :tenant_id"
            params["tenant_id"] = tenant_id
        if user_id:
            sql += " AND s.user_id = :user_id"
            params["user_id"] = user_id
        if session_id:
            sql += " AND m.session_id = :session_id"
            params["session_id"] = session_id

        sql += " ORDER BY m.ts ASC LIMIT :limit OFFSET :offset"
        params["limit"] = limit
        params["offset"] = offset

        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def list_expired_session_ids(self, retention_days: int = 90) -> List[str]:
        from datetime import datetime, timedelta

        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                """
                SELECT session_id FROM sessions
                WHERE status != 'active'
                  AND (
                    (ended_at IS NOT NULL AND ended_at < :cutoff)
                    OR (ended_at IS NULL AND started_at < :cutoff)
                  )
                """,
                {"cutoff": cutoff},
            )
            rows = await cursor.fetchall()
            return [dict(row)["session_id"] for row in rows]

    async def purge_expired_sessions(self, retention_days: int = 90) -> int:
        from datetime import datetime, timedelta

        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        expired_filter = """
            session_id IN (
                SELECT session_id FROM sessions
                WHERE status != 'active'
                  AND (
                    (ended_at IS NOT NULL AND ended_at < :cutoff)
                    OR (ended_at IS NULL AND started_at < :cutoff)
                  )
            )
        """
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                f"DELETE FROM messages_fts WHERE {expired_filter}",
                {"cutoff": cutoff},
            )
            await cursor.execute(
                f"DELETE FROM messages WHERE {expired_filter}",
                {"cutoff": cutoff},
            )
            msg_count = cursor.rowcount
            await cursor.execute(
                f"DELETE FROM tool_calls WHERE {expired_filter}",
                {"cutoff": cutoff},
            )
            await cursor.execute(
                f"""
                DELETE FROM sessions
                WHERE status != 'active'
                  AND (
                    (ended_at IS NOT NULL AND ended_at < :cutoff)
                    OR (ended_at IS NULL AND started_at < :cutoff)
                  )
                """,
                {"cutoff": cutoff},
            )
            session_count = cursor.rowcount
        return int(msg_count or 0) + int(session_count or 0)

    async def anonymize_user_data(self, tenant_id: str, user_id: str) -> int:
        from agent_platform.memory.adapters.compliance_utils import (
            append_audit_log,
            content_sha256,
            record_message_audit_before_redact,
        )

        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                """
                SELECT m.message_id, m.session_id, m.content
                FROM messages m
                JOIN sessions s ON m.session_id = s.session_id
                WHERE s.tenant_id = :tenant_id AND s.user_id = :user_id
                  AND (m.redacted IS NULL OR m.redacted = 0)
                  AND m.content IS NOT NULL AND m.content <> '[redacted]'
                """,
                {"tenant_id": tenant_id, "user_id": user_id},
            )
            msg_rows = [dict(r) for r in await cursor.fetchall()]
            await record_message_audit_before_redact(
                self, tenant_id, user_id, msg_rows
            )
            count = 0
            for row in msg_rows:
                digest = content_sha256(str(row.get("content") or ""))
                await cursor.execute(
                    """
                    UPDATE messages
                    SET content = '[redacted]', redacted = 1, content_hash = :digest
                    WHERE message_id = :message_id
                    """,
                    {"digest": digest, "message_id": row["message_id"]},
                )
                count += cursor.rowcount
            await cursor.execute(
                """
                DELETE FROM messages_fts
                WHERE message_id IN (
                    SELECT message_id FROM messages
                    WHERE session_id IN (
                        SELECT session_id FROM sessions
                        WHERE tenant_id = :tenant_id AND user_id = :user_id
                    )
                )
                """,
                {"tenant_id": tenant_id, "user_id": user_id},
            )
            await cursor.execute(
                """
                SELECT call_id, session_id, result_summary
                FROM tool_calls
                WHERE session_id IN (
                    SELECT session_id FROM sessions
                    WHERE tenant_id = :tenant_id AND user_id = :user_id
                )
                  AND result_summary IS NOT NULL
                  AND result_summary <> '[redacted]'
                """,
                {"tenant_id": tenant_id, "user_id": user_id},
            )
            tool_rows = [dict(r) for r in await cursor.fetchall()]
            for row in tool_rows:
                summary = row.get("result_summary") or ""
                digest = content_sha256(str(summary))
                await append_audit_log(
                    self,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    resource_type="tool_call",
                    resource_id=str(row.get("call_id") or ""),
                    content_hash=digest,
                    action="anonymize",
                    meta={"session_id": row.get("session_id")},
                )
                await cursor.execute(
                    """
                    UPDATE tool_calls
                    SET result_summary = '[redacted]', result_hash = :digest
                    WHERE call_id = :call_id
                    """,
                    {"digest": digest, "call_id": row["call_id"]},
                )
        return int(count or 0)

    async def insert_compliance_audit_log(self, data: dict) -> str:
        columns = ", ".join(k for k in data.keys() if k != "id")
        placeholders = ", ".join(f":{k}" for k in data.keys() if k != "id")
        query = f"""
            INSERT INTO compliance_audit_log ({columns})
            VALUES ({placeholders})
        """
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(query, data)
            return str(cursor.lastrowid)

    async def delete_compliance_audit_logs(
        self,
        *,
        tenant_id: str,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
    ) -> int:
        conditions = ["tenant_id = :tenant_id"]
        params: dict = {"tenant_id": tenant_id}
        if user_id:
            conditions.append("user_id = :user_id")
            params["user_id"] = user_id
        if resource_type:
            conditions.append("resource_type = :resource_type")
            params["resource_type"] = resource_type
        sql = f"DELETE FROM compliance_audit_log WHERE {' AND '.join(conditions)}"
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(sql, params)
            return int(cursor.rowcount or 0)

    async def list_cold_archive_sessions(
        self,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        conditions: List[str] = []
        params: dict = {"limit": limit, "offset": offset}
        if tenant_id:
            conditions.append("tenant_id = :tenant_id")
            params["tenant_id"] = tenant_id
        if user_id:
            conditions.append("user_id = :user_id")
            params["user_id"] = user_id
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT session_id, tenant_id, user_id, object_key, checksum, archived_at
            FROM cold_archive_sessions{where}
            ORDER BY archived_at DESC
            LIMIT :limit OFFSET :offset
        """
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def count_cold_archive_search_rows(self, session_id: str) -> int:
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                "SELECT COUNT(*) AS c FROM cold_archive_search WHERE session_id = :session_id",
                {"session_id": session_id},
            )
            row = await cursor.fetchone()
            return int(dict(row)["c"]) if row else 0

    async def insert_cold_archive_index(self, data: dict) -> str:
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{k}" for k in data.keys())
        query = f"""
            INSERT OR REPLACE INTO cold_archive_sessions ({columns})
            VALUES ({placeholders})
        """
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(query, data)
        return str(data["archive_id"])

    async def get_cold_archive(self, session_id: str) -> Optional[dict]:
        return await self.select_one(
            "cold_archive_sessions",
            [
                "archive_id",
                "session_id",
                "tenant_id",
                "user_id",
                "object_key",
                "message_count",
                "tool_call_count",
                "payload_bytes",
                "checksum",
                "started_at",
                "archived_at",
            ],
            {"session_id": session_id},
        )

    async def list_cold_archives(
        self,
        tenant_id: str,
        user_id: str,
        limit: int = 20,
    ) -> List[dict]:
        return await self.select_many(
            "cold_archive_sessions",
            [
                "archive_id",
                "session_id",
                "tenant_id",
                "user_id",
                "object_key",
                "message_count",
                "tool_call_count",
                "payload_bytes",
                "started_at",
                "archived_at",
            ],
            where={"tenant_id": tenant_id, "user_id": user_id},
            order_by="archived_at DESC",
            limit=limit,
        )

    async def delete_cold_archive_index(self, session_id: str) -> int:
        await self.delete_cold_archive_search(session_id)
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                "DELETE FROM cold_archive_sessions WHERE session_id = :session_id",
                {"session_id": session_id},
            )
            return cursor.rowcount

    async def insert_cold_archive_search_rows(self, rows: List[dict]) -> int:
        if not rows:
            return 0
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            for row in rows:
                await cursor.execute(
                    """
                    INSERT OR REPLACE INTO cold_archive_search
                    (record_id, session_id, tenant_id, user_id, role, content, ts, record_type)
                    VALUES (:record_id, :session_id, :tenant_id, :user_id, :role, :content, :ts, :record_type)
                    """,
                    row,
                )
                await cursor.execute(
                    "DELETE FROM cold_archive_fts WHERE record_id = :record_id",
                    {"record_id": row["record_id"]},
                )
                content = (row.get("content") or "").strip()
                if content:
                    await cursor.execute(
                        """
                        INSERT INTO cold_archive_fts
                        (content, record_id, session_id, tenant_id, user_id)
                        VALUES (:content, :record_id, :session_id, :tenant_id, :user_id)
                        """,
                        {
                            "content": content,
                            "record_id": row["record_id"],
                            "session_id": row["session_id"],
                            "tenant_id": row["tenant_id"],
                            "user_id": row["user_id"],
                        },
                    )
        return len(rows)

    async def delete_cold_archive_search(self, session_id: str) -> int:
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                """
                DELETE FROM cold_archive_fts WHERE record_id IN (
                    SELECT record_id FROM cold_archive_search WHERE session_id = :session_id
                )
                """,
                {"session_id": session_id},
            )
            await cursor.execute(
                "DELETE FROM cold_archive_search WHERE session_id = :session_id",
                {"session_id": session_id},
            )
            return cursor.rowcount

    async def delete_cold_archive_search_for_user(
        self, tenant_id: str, user_id: str
    ) -> int:
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                """
                DELETE FROM cold_archive_fts WHERE record_id IN (
                    SELECT record_id FROM cold_archive_search
                    WHERE tenant_id = :tenant_id AND user_id = :user_id
                )
                """,
                {"tenant_id": tenant_id, "user_id": user_id},
            )
            await cursor.execute(
                """
                DELETE FROM cold_archive_search
                WHERE tenant_id = :tenant_id AND user_id = :user_id
                """,
                {"tenant_id": tenant_id, "user_id": user_id},
            )
            return cursor.rowcount

    async def search_cold_archive_messages(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        *,
        session_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[dict]:
        fts_query = self._build_fts_query(query)
        if fts_query == '""':
            return []

        conditions = ["cold_archive_fts MATCH :fts_query", "c.tenant_id = :tenant_id", "c.user_id = :user_id"]
        params: dict = {
            "fts_query": fts_query,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "limit": limit,
        }
        if session_id:
            conditions.append("c.session_id = :session_id")
            params["session_id"] = session_id

        where = " AND ".join(conditions)
        sql = f"""
            SELECT c.record_id AS message_id, c.session_id, c.role, c.content, c.ts,
                   c.record_type, bm25(cold_archive_fts) AS rank_score
            FROM cold_archive_search c
            JOIN cold_archive_fts f ON c.record_id = f.record_id
            WHERE {where}
            ORDER BY rank_score ASC, c.ts DESC
            LIMIT :limit
        """
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(sql, params)
            rows = await cursor.fetchall()

        hits: List[dict] = []
        for row in rows:
            item = dict(row)
            rank = float(item.pop("rank_score", 0.0))
            score = 1.0 / (1.0 + max(rank, 0.0))
            hits.append(
                {
                    "message_id": item.get("message_id"),
                    "session_id": item.get("session_id"),
                    "role": item.get("role", "user"),
                    "content": item.get("content", ""),
                    "ts": item.get("ts", ""),
                    "score": score,
                    "source": "cold",
                }
            )
        return hits

    async def delete_online_session(self, session_id: str) -> None:
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                "DELETE FROM messages_fts WHERE session_id = :session_id",
                {"session_id": session_id},
            )
            await cursor.execute(
                "DELETE FROM messages WHERE session_id = :session_id",
                {"session_id": session_id},
            )
            await cursor.execute(
                "DELETE FROM tool_calls WHERE session_id = :session_id",
                {"session_id": session_id},
            )
            await cursor.execute(
                "DELETE FROM sessions WHERE session_id = :session_id",
                {"session_id": session_id},
            )

    async def insert_skill_run(self, data: dict) -> str:
        return await self.insert("skill_runs", data)

    async def delete_skill_runs_for_user(self, tenant_id: str, user_id: str) -> int:
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                """
                DELETE FROM skill_runs
                WHERE tenant_id = :tenant_id AND user_id = :user_id
                """,
                {"tenant_id": tenant_id, "user_id": user_id},
            )
            return cursor.rowcount or 0

    async def delete_skill_runs_for_tenant(self, tenant_id: str) -> int:
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(
                "DELETE FROM skill_runs WHERE tenant_id = :tenant_id",
                {"tenant_id": tenant_id},
            )
            return cursor.rowcount or 0

    async def list_skill_runs(
        self,
        *,
        tenant_id: str,
        user_id: Optional[str] = None,
        skill_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[dict]:
        where: dict = {"tenant_id": tenant_id}
        if user_id:
            where["user_id"] = user_id
        if skill_id:
            where["skill_id"] = skill_id
        rows = await self.select_many(
            "skill_runs",
            [
                "run_id",
                "skill_id",
                "tenant_id",
                "user_id",
                "session_id",
                "trace_id",
                "success",
                "steps_executed",
                "error",
                "ts",
            ],
            where=where,
            order_by="ts DESC",
            limit=limit,
            offset=offset,
        )
        return [dict(r) for r in rows]
    
    async def get_hot_memory_doc(
        self, tenant_id: str, doc_kind: str, user_id: str = ""
    ) -> Optional[str]:
        row = await self.select_one(
            "hot_memory_docs",
            ["content"],
            {"tenant_id": tenant_id, "doc_kind": doc_kind, "user_id": user_id or ""},
        )
        return (row or {}).get("content")

    async def upsert_hot_memory_doc(
        self,
        tenant_id: str,
        doc_kind: str,
        content: str,
        *,
        user_id: str = "",
    ) -> None:
        from datetime import datetime

        uid = user_id or ""
        ts = datetime.now().isoformat()
        existing = await self.select_one(
            "hot_memory_docs",
            ["tenant_id"],
            {"tenant_id": tenant_id, "doc_kind": doc_kind, "user_id": uid},
        )
        payload = {
            "tenant_id": tenant_id,
            "doc_kind": doc_kind,
            "user_id": uid,
            "content": content,
            "updated_at": ts,
        }
        if existing:
            await self.update(
                "hot_memory_docs",
                {"content": content, "updated_at": ts},
                {"tenant_id": tenant_id, "doc_kind": doc_kind, "user_id": uid},
            )
        else:
            await self.insert("hot_memory_docs", payload)

    async def delete_hot_memory_doc(
        self, tenant_id: str, doc_kind: str, user_id: str = ""
    ) -> None:
        query = (
            "DELETE FROM hot_memory_docs WHERE tenant_id = :tenant_id "
            "AND doc_kind = :doc_kind AND user_id = :user_id"
        )
        params = {
            "tenant_id": tenant_id,
            "doc_kind": doc_kind,
            "user_id": user_id or "",
        }
        async with self._get_connection() as conn:
            await conn.execute(query, params)
            await conn.commit()

    async def list_hot_memory_user_ids(self, tenant_id: str) -> List[str]:
        rows = await self.select_many(
            "hot_memory_docs",
            ["user_id"],
            where={"tenant_id": tenant_id, "doc_kind": "user"},
        )
        return sorted({r["user_id"] for r in rows if r.get("user_id")})

    async def health(self) -> dict:
        try:
            async with self._get_connection() as conn:
                await conn.execute("SELECT 1")
            return {
                "status": "healthy",
                "type": "async_sqlite",
                "pool_size": self._pool_size,
                "available": len(self._pool)
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def close(self) -> None:
        for conn in self._pool:
            await conn.close()
        self._pool.clear()
        self._logger.info("Connection pool closed")
