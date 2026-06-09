import asyncio
import logging
from typing import Optional, List, Any, Dict
from contextlib import asynccontextmanager
from datetime import datetime
import time

from core.ports.storage.relational import RelationalPort


class PostgreSQLAdapter:
    """PostgreSQL 关系型数据库适配器
    
    企业级特性：
    - asyncpg 连接池
    - 熔断器保护
    - 重试机制
    - 健康检查
    - 慢查询日志
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "agents",
        user: str = "postgres",
        password: str = "",
        pool_size: int = 10,
        max_overflow: int = 5,
        command_timeout: float = 30.0,
        slow_query_threshold: float = 1.0,
        enable_circuit_breaker: bool = True,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ):
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._command_timeout = command_timeout
        self._slow_query_threshold = slow_query_threshold
        
        self._pool = None
        self._logger = logging.getLogger("storage.postgresql")
        
        self._enable_cb = enable_circuit_breaker
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._last_failure_time = 0
        self._circuit_open = False
    
    async def _init_pool(self) -> None:
        if self._pool is not None:
            return
        
        try:
            import asyncpg
            
            self._pool = await asyncpg.create_pool(
                host=self._host,
                port=self._port,
                database=self._database,
                user=self._user,
                password=self._password,
                min_size=1,
                max_size=self._pool_size + self._max_overflow,
                command_timeout=self._command_timeout,
            )
            
            self._logger.info(
                f"PostgreSQL pool initialized: {self._host}:{self._port}/{self._database}"
            )
            
            await self._init_tables()
            
        except Exception as e:
            self._logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise
    
    async def _init_tables(self) -> None:
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id VARCHAR(64) PRIMARY KEY,
            user_id VARCHAR(64) NOT NULL,
            tenant_id VARCHAR(64) NOT NULL,
            channel VARCHAR(32),
            started_at TIMESTAMP NOT NULL,
            ended_at TIMESTAMP,
            status VARCHAR(16) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS messages (
            message_id VARCHAR(64) PRIMARY KEY,
            session_id VARCHAR(64) NOT NULL REFERENCES sessions(session_id),
            role VARCHAR(16) NOT NULL,
            content TEXT,
            ts TIMESTAMP NOT NULL,
            token_count INTEGER DEFAULT 0,
            redacted BOOLEAN DEFAULT FALSE,
            metadata_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS tool_calls (
            call_id VARCHAR(64) PRIMARY KEY,
            session_id VARCHAR(64) NOT NULL REFERENCES sessions(session_id),
            tool_name VARCHAR(128) NOT NULL,
            args_hash VARCHAR(64),
            result_summary TEXT,
            status VARCHAR(16),
            latency_ms INTEGER,
            ts TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts);
        CREATE INDEX IF NOT EXISTS idx_messages_content_fts
            ON messages USING gin (to_tsvector('simple', coalesce(content, '')));
        CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(tenant_id, user_id);

        CREATE TABLE IF NOT EXISTS cold_archive_sessions (
            archive_id VARCHAR(64) PRIMARY KEY,
            session_id VARCHAR(64) NOT NULL UNIQUE,
            tenant_id VARCHAR(64) NOT NULL,
            user_id VARCHAR(64) NOT NULL,
            object_key TEXT NOT NULL,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            payload_bytes INTEGER DEFAULT 0,
            checksum VARCHAR(128),
            started_at TIMESTAMP,
            archived_at TIMESTAMP NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cold_archive_tenant_user
            ON cold_archive_sessions(tenant_id, user_id);

        CREATE TABLE IF NOT EXISTS cold_archive_search (
            record_id VARCHAR(64) PRIMARY KEY,
            session_id VARCHAR(64) NOT NULL,
            tenant_id VARCHAR(64) NOT NULL,
            user_id VARCHAR(64) NOT NULL,
            role VARCHAR(16) NOT NULL,
            content TEXT NOT NULL,
            ts TIMESTAMP,
            record_type VARCHAR(16) DEFAULT 'message'
        );
        CREATE INDEX IF NOT EXISTS idx_cas_session ON cold_archive_search(session_id);
        CREATE INDEX IF NOT EXISTS idx_cas_tenant_user
            ON cold_archive_search(tenant_id, user_id);
        CREATE INDEX IF NOT EXISTS idx_cas_content_fts
            ON cold_archive_search
            USING gin (to_tsvector('simple', coalesce(content, '')));

        CREATE TABLE IF NOT EXISTS compliance_audit_log (
            id SERIAL PRIMARY KEY,
            tenant_id VARCHAR(64) NOT NULL,
            user_id VARCHAR(64) NOT NULL,
            resource_type VARCHAR(32) NOT NULL,
            resource_id VARCHAR(64) NOT NULL,
            content_hash VARCHAR(128) NOT NULL,
            action VARCHAR(32) NOT NULL,
            ts TIMESTAMP NOT NULL,
            meta_json JSONB
        );
        CREATE INDEX IF NOT EXISTS idx_audit_tenant_user
            ON compliance_audit_log(tenant_id, user_id);

        CREATE TABLE IF NOT EXISTS graph_checkpoints (
            checkpoint_id VARCHAR(64) PRIMARY KEY,
            thread_id VARCHAR(128) NOT NULL,
            tenant_id VARCHAR(64) NOT NULL,
            session_id VARCHAR(128),
            checkpoint_ns VARCHAR(64) DEFAULT '',
            parent_id VARCHAR(64),
            state_json JSONB NOT NULL,
            metadata_json JSONB,
            created_at TIMESTAMP NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_gc_thread
            ON graph_checkpoints(thread_id, tenant_id);

        CREATE TABLE IF NOT EXISTS skill_runs (
            run_id VARCHAR(64) PRIMARY KEY,
            skill_id VARCHAR(128) NOT NULL,
            tenant_id VARCHAR(64) NOT NULL,
            user_id VARCHAR(64) NOT NULL,
            session_id VARCHAR(128),
            trace_id VARCHAR(64),
            success INTEGER NOT NULL DEFAULT 0,
            steps_executed INTEGER DEFAULT 0,
            error TEXT,
            inputs_json TEXT,
            outputs_json TEXT,
            ts TIMESTAMP NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_skill_runs_tenant_user
            ON skill_runs(tenant_id, user_id);
        CREATE INDEX IF NOT EXISTS idx_skill_runs_skill
            ON skill_runs(skill_id);

        CREATE TABLE IF NOT EXISTS hot_memory_docs (
            tenant_id VARCHAR(64) NOT NULL,
            doc_kind VARCHAR(32) NOT NULL,
            user_id VARCHAR(64) NOT NULL DEFAULT '',
            content TEXT NOT NULL DEFAULT '',
            updated_at TIMESTAMP NOT NULL,
            PRIMARY KEY (tenant_id, doc_kind, user_id)
        );
        """
        
        await self.execute(create_tables_sql)
        await self._migrate_schema()
    
    async def _migrate_schema(self) -> None:
        for stmt in (
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS content_hash VARCHAR(128)",
            "ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS result_hash VARCHAR(128)",
        ):
            try:
                await self.execute(stmt)
            except Exception:
                pass
    
    def _check_circuit_breaker(self) -> bool:
        if not self._enable_cb:
            return True
        
        if self._circuit_open:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                self._circuit_open = False
                self._failure_count = 0
                self._logger.info("Circuit breaker closed, attempting recovery")
                return True
            return False
        
        return True
    
    def _record_failure(self) -> None:
        if not self._enable_cb:
            return
        
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self._failure_threshold:
            self._circuit_open = True
            self._logger.warning(
                f"Circuit breaker opened after {self._failure_count} failures"
            )
    
    def _record_success(self) -> None:
        if self._enable_cb:
            self._failure_count = 0
    
    @asynccontextmanager
    async def _get_connection(self):
        await self._init_pool()
        
        if not self._check_circuit_breaker():
            raise Exception("Circuit breaker is open")
        
        async with self._pool.acquire() as conn:
            try:
                yield conn
                self._record_success()
            except Exception as e:
                self._record_failure()
                raise
    
    async def execute(self, query: str, params: Optional[Dict] = None) -> Any:
        start_time = time.time()
        
        try:
            async with self._get_connection() as conn:
                if params:
                    result = await conn.execute(query, *params.values() if isinstance(params, dict) else params)
                else:
                    result = await conn.execute(query)
                
                elapsed = time.time() - start_time
                if elapsed > self._slow_query_threshold:
                    self._logger.warning(
                        f"Slow query ({elapsed:.3f}s): {query[:100]}..."
                    )
                
                return result
                
        except Exception as e:
            self._logger.error(f"Execute failed: {e}")
            raise
    
    async def execute_batch(self, queries: List[tuple]) -> List[Any]:
        results = []
        async with self._get_connection() as conn:
            async with conn.transaction():
                for query, params in queries:
                    if params:
                        result = await conn.execute(query, *params.values() if isinstance(params, dict) else params)
                    else:
                        result = await conn.execute(query)
                    results.append(result)
        return results
    
    async def insert(self, table: str, data: Dict) -> str:
        columns = list(data.keys())
        placeholders = [f"${i+1}" for i in range(len(columns))]
        
        query = f"""
            INSERT INTO {table} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            RETURNING {columns[0]}
        """
        
        async with self._get_connection() as conn:
            result = await conn.fetchval(query, *data.values())
            return str(result)
    
    async def update(self, table: str, data: Dict, where: Dict) -> int:
        set_clause = ", ".join([f"{k} = ${i+1}" for i, k in enumerate(data.keys())])
        where_clause = " AND ".join([f"{k} = ${len(data)+i+1}" for i, k in enumerate(where.keys())])
        
        query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        params = list(data.values()) + list(where.values())
        
        async with self._get_connection() as conn:
            result = await conn.execute(query, *params)
            return int(result.split()[-1])
    
    async def delete(self, table: str, where: Dict) -> int:
        where_clause = " AND ".join([f"{k} = ${i+1}" for i, k in enumerate(where.keys())])
        query = f"DELETE FROM {table} WHERE {where_clause}"
        
        async with self._get_connection() as conn:
            result = await conn.execute(query, *where.values())
            return int(result.split()[-1])
    
    async def select_one(
        self,
        table: str,
        columns: List[str],
        where: Dict
    ) -> Optional[Dict]:
        cols = ", ".join(columns)
        where_clause = " AND ".join([f"{k} = ${i+1}" for i, k in enumerate(where.keys())])
        
        query = f"SELECT {cols} FROM {table} WHERE {where_clause} LIMIT 1"
        
        async with self._get_connection() as conn:
            row = await conn.fetchrow(query, *where.values())
            
            if row:
                return dict(row)
            return None
    
    async def select_many(
        self,
        table: str,
        columns: List[str],
        where: Optional[Dict] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        cols = ", ".join(columns)
        query = f"SELECT {cols} FROM {table}"
        params = []
        
        if where:
            where_clause = " AND ".join([f"{k} = ${i+1}" for i, k in enumerate(where.keys())])
            query += f" WHERE {where_clause}"
            params = list(where.values())
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"

        if offset:
            query += f" OFFSET {offset}"
        
        async with self._get_connection() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def begin_transaction(self) -> Any:
        await self._init_pool()
        conn = await self._pool.acquire()
        tr = conn.transaction()
        await tr.start()
        return (conn, tr)
    
    async def commit(self, transaction: tuple) -> None:
        conn, tr = transaction
        await tr.commit()
        await self._pool.release(conn)
    
    async def rollback(self, transaction: tuple) -> None:
        conn, tr = transaction
        await tr.rollback()
        await self._pool.release(conn)
    
    async def health_check(self) -> Dict[str, Any]:
        try:
            async with self._get_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                
                pool_size = self._pool.get_size() if self._pool else 0
                idle_size = self._pool.get_idle_size() if self._pool else 0
                
                return {
                    "status": "healthy",
                    "database": self._database,
                    "pool_size": pool_size,
                    "idle_connections": idle_size,
                    "circuit_breaker": "closed" if not self._circuit_open else "open",
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker": "closed" if not self._circuit_open else "open",
            }
    
    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._logger.info("PostgreSQL pool closed")

    # --- L2 Session Archive (parity with AsyncSQLiteRelationalAdapter) ---

    async def upsert_session(self, data: dict) -> None:
        query = """
            INSERT INTO sessions (
                session_id, user_id, tenant_id, channel, started_at, status
            ) VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (session_id) DO NOTHING
        """
        async with self._get_connection() as conn:
            await conn.execute(
                query,
                data["session_id"],
                data["user_id"],
                data["tenant_id"],
                data.get("channel"),
                data["started_at"],
                data.get("status", "active"),
            )

    async def end_session(self, session_id: str, status: str = "closed") -> None:
        await self.update(
            "sessions",
            {"ended_at": datetime.now(), "status": status},
            {"session_id": session_id},
        )

    async def list_sessions(
        self,
        tenant_id: str,
        user_id: str,
        limit: int = 20,
    ) -> List[Dict]:
        return await self.select_many(
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
            where={"tenant_id": tenant_id, "user_id": user_id},
            order_by="started_at DESC",
            limit=limit,
        )

    async def insert_message(self, data: dict) -> str:
        columns = list(data.keys())
        placeholders = [f"${i + 1}" for i in range(len(columns))]
        updates = ", ".join(
            f"{col} = EXCLUDED.{col}"
            for col in columns
            if col != "message_id"
        )
        query = f"""
            INSERT INTO messages ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            ON CONFLICT (message_id) DO UPDATE SET {updates}
            RETURNING message_id
        """
        async with self._get_connection() as conn:
            message_id = await conn.fetchval(query, *data.values())
            return str(message_id)

    async def insert_tool_call(self, data: dict) -> str:
        columns = list(data.keys())
        placeholders = [f"${i + 1}" for i in range(len(columns))]
        query = f"""
            INSERT INTO tool_calls ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            ON CONFLICT (call_id) DO NOTHING
            RETURNING call_id
        """
        async with self._get_connection() as conn:
            call_id = await conn.fetchval(query, *data.values())
            return str(call_id or data["call_id"])

    @staticmethod
    def _build_tsquery_terms(query: str) -> str:
        import re

        tokens = [t for t in re.split(r"\s+", query.strip()) if t]
        if not tokens:
            return ""
        return " | ".join(tokens)

    async def _search_messages_fts(
        self,
        session_id: Optional[str],
        user_id: Optional[str],
        tenant_id: Optional[str],
        query: str,
        limit: int,
    ) -> List[Dict]:
        tsquery = self._build_tsquery_terms(query)
        if not tsquery:
            return []

        conditions = [
            "to_tsvector('simple', coalesce(m.content, '')) @@ to_tsquery('simple', $1)",
            "(m.redacted IS NULL OR m.redacted = FALSE)",
        ]
        params: List[Any] = [tsquery]
        idx = 2

        if session_id:
            conditions.append(f"m.session_id = ${idx}")
            params.append(session_id)
            idx += 1
        if user_id:
            conditions.append(f"s.user_id = ${idx}")
            params.append(user_id)
            idx += 1
        if tenant_id:
            conditions.append(f"s.tenant_id = ${idx}")
            params.append(tenant_id)
            idx += 1

        params.append(limit)
        where = " AND ".join(conditions)
        sql = f"""
            SELECT m.* FROM messages m
            JOIN sessions s ON m.session_id = s.session_id
            WHERE {where}
            ORDER BY ts_rank(
                to_tsvector('simple', coalesce(m.content, '')),
                to_tsquery('simple', $1)
            ) DESC, m.ts DESC
            LIMIT ${idx}
        """
        async with self._get_connection() as conn:
            rows = await conn.fetch(sql, *params)
            return [dict(row) for row in rows]

    async def _search_messages_like(
        self,
        session_id: Optional[str],
        user_id: Optional[str],
        tenant_id: Optional[str],
        query: Optional[str],
        limit: int,
    ) -> List[Dict]:
        use_join = bool(user_id or tenant_id)
        if use_join:
            sql = """
                SELECT m.* FROM messages m
                JOIN sessions s ON m.session_id = s.session_id
                WHERE (m.redacted IS NULL OR m.redacted = FALSE)
            """
        else:
            sql = """
                SELECT * FROM messages
                WHERE (redacted IS NULL OR redacted = FALSE)
            """

        params: List[Any] = []
        idx = 1

        if session_id:
            col = "m.session_id" if use_join else "session_id"
            sql += f" AND {col} = ${idx}"
            params.append(session_id)
            idx += 1
        if user_id:
            sql += f" AND s.user_id = ${idx}"
            params.append(user_id)
            idx += 1
        if tenant_id:
            sql += f" AND s.tenant_id = ${idx}"
            params.append(tenant_id)
            idx += 1
        if query:
            col = "m.content" if use_join else "content"
            sql += f" AND {col} ILIKE ${idx}"
            params.append(f"%{query}%")
            idx += 1

        order_col = "m.ts" if use_join else "ts"
        params.append(limit)
        sql += f" ORDER BY {order_col} DESC LIMIT ${idx}"

        async with self._get_connection() as conn:
            rows = await conn.fetch(sql, *params)
            return [dict(row) for row in rows]

    async def search_messages(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        if query:
            try:
                rows = await self._search_messages_fts(
                    session_id, user_id, tenant_id, query, limit
                )
                if rows:
                    return rows
            except Exception as e:
                self._logger.debug("PG FTS search fallback to ILIKE: %s", e)
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
    ) -> List[Dict]:
        if not query:
            return []
        tokens = [t for t in query.split() if t]
        params: List[Any] = []
        idx = 1
        token_parts = []
        for token in tokens:
            token_parts.append(
                f"(t.result_summary ILIKE ${idx} OR t.tool_name ILIKE ${idx})"
            )
            params.append(f"%{token}%")
            idx += 1
        conditions = []
        if token_parts:
            conditions.append(f"({' AND '.join(token_parts)})")
        if session_id:
            conditions.append(f"t.session_id = ${idx}")
            params.append(session_id)
            idx += 1
        if user_id:
            conditions.append(f"s.user_id = ${idx}")
            params.append(user_id)
            idx += 1
        if tenant_id:
            conditions.append(f"s.tenant_id = ${idx}")
            params.append(tenant_id)
            idx += 1

        where = " AND ".join(conditions)
        params.append(limit)
        sql = f"""
            SELECT t.call_id AS message_id, t.session_id, 'tool' AS role,
                   (t.tool_name || ': ' || COALESCE(t.result_summary, '')) AS content,
                   t.ts, s.tenant_id, s.user_id
            FROM tool_calls t
            JOIN sessions s ON t.session_id = s.session_id
            WHERE {where}
            ORDER BY t.ts DESC
            LIMIT ${idx}
        """
        async with self._get_connection() as conn:
            rows = await conn.fetch(sql, *params)
            return [dict(row) for row in rows]

    async def list_messages_for_reindex(
        self,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List[Dict]:
        sql = """
            SELECT m.message_id, m.session_id, m.role, m.content, m.ts,
                   m.redacted, s.tenant_id, s.user_id
            FROM messages m
            JOIN sessions s ON m.session_id = s.session_id
            WHERE (m.redacted IS NULL OR m.redacted = FALSE)
              AND m.content IS NOT NULL
              AND m.content <> ''
              AND m.content <> '[redacted]'
        """
        params: List[Any] = []
        idx = 1

        if tenant_id:
            sql += f" AND s.tenant_id = ${idx}"
            params.append(tenant_id)
            idx += 1
        if user_id:
            sql += f" AND s.user_id = ${idx}"
            params.append(user_id)
            idx += 1
        if session_id:
            sql += f" AND m.session_id = ${idx}"
            params.append(session_id)
            idx += 1

        params.extend([limit, offset])
        sql += f" ORDER BY m.ts ASC LIMIT ${idx} OFFSET ${idx + 1}"

        async with self._get_connection() as conn:
            rows = await conn.fetch(sql, *params)
            return [dict(row) for row in rows]

    async def list_expired_session_ids(self, retention_days: int = 90) -> List[str]:
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=retention_days)
        async with self._get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id FROM sessions
                WHERE status != 'active'
                  AND (
                    (ended_at IS NOT NULL AND ended_at < $1)
                    OR (ended_at IS NULL AND started_at < $1)
                  )
                """,
                cutoff,
            )
            return [row["session_id"] for row in rows]

    async def purge_expired_sessions(self, retention_days: int = 90) -> int:
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=retention_days)
        expired_subquery = """
            SELECT session_id FROM sessions
            WHERE status != 'active'
              AND (
                (ended_at IS NOT NULL AND ended_at < $1)
                OR (ended_at IS NULL AND started_at < $1)
              )
        """
        async with self._get_connection() as conn:
            msg_result = await conn.execute(
                f"DELETE FROM messages WHERE session_id IN ({expired_subquery})",
                cutoff,
            )
            await conn.execute(
                f"DELETE FROM tool_calls WHERE session_id IN ({expired_subquery})",
                cutoff,
            )
            sess_result = await conn.execute(
                f"""
                DELETE FROM sessions
                WHERE status != 'active'
                  AND (
                    (ended_at IS NOT NULL AND ended_at < $1)
                    OR (ended_at IS NULL AND started_at < $1)
                  )
                """,
                cutoff,
            )
        msg_count = int(msg_result.split()[-1]) if msg_result else 0
        sess_count = int(sess_result.split()[-1]) if sess_result else 0
        return msg_count + sess_count

    async def anonymize_user_data(self, tenant_id: str, user_id: str) -> int:
        from agent_platform.memory.adapters.compliance_utils import (
            append_audit_log,
            content_sha256,
            record_message_audit_before_redact,
        )

        count = 0
        async with self._get_connection() as conn:
            msg_rows = await conn.fetch(
                """
                SELECT m.message_id, m.session_id, m.content
                FROM messages m
                JOIN sessions s ON m.session_id = s.session_id
                WHERE s.tenant_id = $1 AND s.user_id = $2
                  AND (m.redacted IS NULL OR m.redacted = FALSE)
                  AND m.content IS NOT NULL AND m.content <> '[redacted]'
                """,
                tenant_id,
                user_id,
            )
            msg_list = [dict(r) for r in msg_rows]
            await record_message_audit_before_redact(
                self, tenant_id, user_id, msg_list
            )
            for row in msg_list:
                digest = content_sha256(str(row.get("content") or ""))
                res = await conn.execute(
                    """
                    UPDATE messages
                    SET content = '[redacted]', redacted = TRUE, content_hash = $1
                    WHERE message_id = $2
                    """,
                    digest,
                    row["message_id"],
                )
                count += int(res.split()[-1]) if res else 0

            tool_rows = await conn.fetch(
                """
                SELECT call_id, session_id, result_summary
                FROM tool_calls
                WHERE session_id IN (
                    SELECT session_id FROM sessions
                    WHERE tenant_id = $1 AND user_id = $2
                )
                  AND result_summary IS NOT NULL
                  AND result_summary <> '[redacted]'
                """,
                tenant_id,
                user_id,
            )
            for row in tool_rows:
                digest = content_sha256(str(row["result_summary"] or ""))
                await append_audit_log(
                    self,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    resource_type="tool_call",
                    resource_id=str(row["call_id"]),
                    content_hash=digest,
                    action="anonymize",
                    meta={"session_id": row["session_id"]},
                )
                await conn.execute(
                    """
                    UPDATE tool_calls
                    SET result_summary = '[redacted]', result_hash = $1
                    WHERE call_id = $2
                    """,
                    digest,
                    row["call_id"],
                )
        return count

    async def insert_compliance_audit_log(self, data: dict) -> str:
        columns = [k for k in data.keys() if k != "id"]
        placeholders = [f"${i + 1}" for i in range(len(columns))]
        query = f"""
            INSERT INTO compliance_audit_log ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            RETURNING id
        """
        async with self._get_connection() as conn:
            row_id = await conn.fetchval(query, *[data[c] for c in columns])
            return str(row_id)

    async def delete_compliance_audit_logs(
        self,
        *,
        tenant_id: str,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
    ) -> int:
        conditions = ["tenant_id = $1"]
        params: List[Any] = [tenant_id]
        idx = 2
        if user_id:
            conditions.append(f"user_id = ${idx}")
            params.append(user_id)
            idx += 1
        if resource_type:
            conditions.append(f"resource_type = ${idx}")
            params.append(resource_type)
        sql = f"DELETE FROM compliance_audit_log WHERE {' AND '.join(conditions)}"
        async with self._get_connection() as conn:
            result = await conn.execute(sql, *params)
            try:
                return int(str(result).split()[-1])
            except (ValueError, IndexError):
                return 0

    async def list_cold_archive_sessions(
        self,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        conditions: List[str] = []
        params: List[Any] = []
        idx = 1
        if tenant_id:
            conditions.append(f"tenant_id = ${idx}")
            params.append(tenant_id)
            idx += 1
        if user_id:
            conditions.append(f"user_id = ${idx}")
            params.append(user_id)
            idx += 1
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])
        sql = f"""
            SELECT session_id, tenant_id, user_id, object_key, checksum, archived_at
            FROM cold_archive_sessions{where}
            ORDER BY archived_at DESC
            LIMIT ${idx} OFFSET ${idx + 1}
        """
        async with self._get_connection() as conn:
            rows = await conn.fetch(sql, *params)
            return [dict(r) for r in rows]

    async def count_cold_archive_search_rows(self, session_id: str) -> int:
        async with self._get_connection() as conn:
            return int(
                await conn.fetchval(
                    "SELECT COUNT(*) FROM cold_archive_search WHERE session_id = $1",
                    session_id,
                )
                or 0
            )

    async def insert_cold_archive_index(self, data: dict) -> str:
        columns = list(data.keys())
        placeholders = [f"${i + 1}" for i in range(len(columns))]
        updates = ", ".join(
            f"{col} = EXCLUDED.{col}"
            for col in columns
            if col != "archive_id"
        )
        query = f"""
            INSERT INTO cold_archive_sessions ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            ON CONFLICT (session_id) DO UPDATE SET {updates}
            RETURNING archive_id
        """
        async with self._get_connection() as conn:
            archive_id = await conn.fetchval(query, *data.values())
            return str(archive_id)

    async def get_cold_archive(self, session_id: str) -> Optional[Dict]:
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
    ) -> List[Dict]:
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
        return await self.delete("cold_archive_sessions", {"session_id": session_id})

    async def insert_cold_archive_search_rows(self, rows: List[Dict]) -> int:
        if not rows:
            return 0
        async with self._get_connection() as conn:
            for row in rows:
                await conn.execute(
                    """
                    INSERT INTO cold_archive_search
                    (record_id, session_id, tenant_id, user_id, role, content, ts, record_type)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (record_id) DO UPDATE SET
                        session_id = EXCLUDED.session_id,
                        tenant_id = EXCLUDED.tenant_id,
                        user_id = EXCLUDED.user_id,
                        role = EXCLUDED.role,
                        content = EXCLUDED.content,
                        ts = EXCLUDED.ts,
                        record_type = EXCLUDED.record_type
                    """,
                    row["record_id"],
                    row["session_id"],
                    row["tenant_id"],
                    row["user_id"],
                    row["role"],
                    row["content"],
                    row.get("ts"),
                    row.get("record_type", "message"),
                )
        return len(rows)

    async def delete_cold_archive_search(self, session_id: str) -> int:
        async with self._get_connection() as conn:
            result = await conn.execute(
                "DELETE FROM cold_archive_search WHERE session_id = $1",
                session_id,
            )
        return int(result.split()[-1]) if result else 0

    async def delete_cold_archive_search_for_user(
        self, tenant_id: str, user_id: str
    ) -> int:
        async with self._get_connection() as conn:
            result = await conn.execute(
                """
                DELETE FROM cold_archive_search
                WHERE tenant_id = $1 AND user_id = $2
                """,
                tenant_id,
                user_id,
            )
        return int(result.split()[-1]) if result else 0

    async def search_cold_archive_messages(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        *,
        session_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        tsquery = self._build_tsquery_terms(query)
        if not tsquery:
            return []

        conditions = [
            "to_tsvector('simple', coalesce(c.content, '')) @@ to_tsquery('simple', $1)",
            "c.tenant_id = $2",
            "c.user_id = $3",
        ]
        params: List[Any] = [tsquery, tenant_id, user_id]
        idx = 4
        if session_id:
            conditions.append(f"c.session_id = ${idx}")
            params.append(session_id)
            idx += 1
        params.append(limit)
        where = " AND ".join(conditions)
        sql = f"""
            SELECT c.record_id AS message_id, c.session_id, c.role, c.content, c.ts,
                   c.record_type,
                   ts_rank(
                       to_tsvector('simple', coalesce(c.content, '')),
                       to_tsquery('simple', $1)
                   ) AS rank_score
            FROM cold_archive_search c
            WHERE {where}
            ORDER BY rank_score DESC, c.ts DESC
            LIMIT ${idx}
        """
        async with self._get_connection() as conn:
            rows = await conn.fetch(sql, *params)

        return [
            {
                "message_id": row["message_id"],
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "ts": str(row["ts"]) if row["ts"] is not None else "",
                "score": float(row["rank_score"] or 0.0),
                "source": "cold",
            }
            for row in rows
        ]

    async def delete_online_session(self, session_id: str) -> None:
        async with self._get_connection() as conn:
            await conn.execute(
                "DELETE FROM messages WHERE session_id = $1",
                session_id,
            )
            await conn.execute(
                "DELETE FROM tool_calls WHERE session_id = $1",
                session_id,
            )
            await conn.execute(
                "DELETE FROM sessions WHERE session_id = $1",
                session_id,
            )

    async def insert_skill_run(self, data: dict) -> str:
        return await self.insert("skill_runs", data)

    async def delete_skill_runs_for_user(self, tenant_id: str, user_id: str) -> int:
        async with self._get_connection() as conn:
            result = await conn.execute(
                """
                DELETE FROM skill_runs
                WHERE tenant_id = $1 AND user_id = $2
                """,
                tenant_id,
                user_id,
            )
        return int(result.split()[-1]) if result else 0

    async def delete_skill_runs_for_tenant(self, tenant_id: str) -> int:
        async with self._get_connection() as conn:
            result = await conn.execute(
                "DELETE FROM skill_runs WHERE tenant_id = $1",
                tenant_id,
            )
        return int(result.split()[-1]) if result else 0

    async def list_skill_runs(
        self,
        *,
        tenant_id: str,
        user_id: Optional[str] = None,
        skill_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict]:
        clauses = ["tenant_id = $1"]
        params: list = [tenant_id]
        idx = 2
        if user_id:
            clauses.append(f"user_id = ${idx}")
            params.append(user_id)
            idx += 1
        if skill_id:
            clauses.append(f"skill_id = ${idx}")
            params.append(skill_id)
            idx += 1
        params.extend([limit, offset])
        sql = f"""
            SELECT run_id, skill_id, tenant_id, user_id, session_id,
                   trace_id, success, steps_executed, error, ts
            FROM skill_runs
            WHERE {' AND '.join(clauses)}
            ORDER BY ts DESC
            LIMIT ${idx} OFFSET ${idx + 1}
        """
        async with self._get_connection() as conn:
            rows = await conn.fetch(sql, *params)
        return [
            {
                "run_id": row["run_id"],
                "skill_id": row["skill_id"],
                "tenant_id": row["tenant_id"],
                "user_id": row["user_id"],
                "session_id": row["session_id"],
                "trace_id": row["trace_id"],
                "success": row["success"],
                "steps_executed": row["steps_executed"],
                "error": row["error"],
                "ts": str(row["ts"]) if row["ts"] is not None else "",
            }
            for row in rows
        ]

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
        if existing:
            await self.update(
                "hot_memory_docs",
                {"content": content, "updated_at": ts},
                {"tenant_id": tenant_id, "doc_kind": doc_kind, "user_id": uid},
            )
        else:
            await self.insert(
                "hot_memory_docs",
                {
                    "tenant_id": tenant_id,
                    "doc_kind": doc_kind,
                    "user_id": uid,
                    "content": content,
                    "updated_at": ts,
                },
            )

    async def delete_hot_memory_doc(
        self, tenant_id: str, doc_kind: str, user_id: str = ""
    ) -> None:
        await self.delete(
            "hot_memory_docs",
            {
                "tenant_id": tenant_id,
                "doc_kind": doc_kind,
                "user_id": user_id or "",
            },
        )

    async def list_hot_memory_user_ids(self, tenant_id: str) -> List[str]:
        rows = await self.select_many(
            "hot_memory_docs",
            ["user_id"],
            where={"tenant_id": tenant_id, "doc_kind": "user"},
        )
        return sorted({r["user_id"] for r in rows if r.get("user_id")})

    async def health(self) -> Dict[str, Any]:
        return await self.health_check()
