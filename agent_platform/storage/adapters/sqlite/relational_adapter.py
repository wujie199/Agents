import asyncio
import aiosqlite
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
        """)
    
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
        limit: Optional[int] = None
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
        
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(query, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    async def search_messages(
        self,
        session_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100
    ) -> List[dict]:
        sql = "SELECT * FROM messages WHERE 1=1"
        params = {}
        
        if session_id:
            sql += " AND session_id = :session_id"
            params["session_id"] = session_id
        
        if query:
            sql += " AND content LIKE :query"
            params["query"] = f"%{query}%"
        
        sql += f" ORDER BY ts DESC LIMIT {limit}"
        
        async with self._get_connection() as conn:
            cursor = await conn.cursor()
            await cursor.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
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
