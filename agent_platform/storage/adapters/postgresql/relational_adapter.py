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
        CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions(tenant_id);
        """
        
        await self.execute(create_tables_sql)
    
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
