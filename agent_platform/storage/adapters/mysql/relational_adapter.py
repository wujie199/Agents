import asyncio
import logging
from typing import Optional, List, Any, Dict
from contextlib import asynccontextmanager
import time

from core.ports.storage.relational import RelationalPort


class MySQLAdapter:
    """MySQL 关系型数据库适配器
    
    企业级特性：
    - aiomysql 连接池
    - 熔断器保护
    - 重试机制
    - 健康检查
    - 慢查询日志
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        database: str = "agents",
        user: str = "root",
        password: str = "",
        pool_size: int = 10,
        max_overflow: int = 5,
        connect_timeout: float = 10.0,
        command_timeout: float = 30.0,
        slow_query_threshold: float = 1.0,
        charset: str = "utf8mb4",
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
        self._connect_timeout = connect_timeout
        self._command_timeout = command_timeout
        self._slow_query_threshold = slow_query_threshold
        self._charset = charset
        
        self._pool = None
        self._logger = logging.getLogger("storage.mysql")
        
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
            import aiomysql
            
            self._pool = await aiomysql.create_pool(
                host=self._host,
                port=self._port,
                db=self._database,
                user=self._user,
                password=self._password,
                minsize=1,
                maxsize=self._pool_size + self._max_overflow,
                connect_timeout=self._connect_timeout,
                charset=self._charset,
                autocommit=False,
            )
            
            self._logger.info(
                f"MySQL pool initialized: {self._host}:{self._port}/{self._database}"
            )
            
            await self._init_tables()
            
        except Exception as e:
            self._logger.error(f"Failed to initialize MySQL pool: {e}")
            raise
    
    async def _init_tables(self) -> None:
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id VARCHAR(64) PRIMARY KEY,
            user_id VARCHAR(64) NOT NULL,
            tenant_id VARCHAR(64) NOT NULL,
            channel VARCHAR(32),
            started_at DATETIME NOT NULL,
            ended_at DATETIME,
            status VARCHAR(16) DEFAULT 'active',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_sessions_tenant (tenant_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        
        CREATE TABLE IF NOT EXISTS messages (
            message_id VARCHAR(64) PRIMARY KEY,
            session_id VARCHAR(64) NOT NULL,
            role VARCHAR(16) NOT NULL,
            content TEXT,
            ts DATETIME NOT NULL,
            token_count INT DEFAULT 0,
            redacted TINYINT DEFAULT 0,
            metadata_json JSON,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_messages_session (session_id),
            INDEX idx_messages_ts (ts),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        
        CREATE TABLE IF NOT EXISTS tool_calls (
            call_id VARCHAR(64) PRIMARY KEY,
            session_id VARCHAR(64) NOT NULL,
            tool_name VARCHAR(128) NOT NULL,
            args_hash VARCHAR(64),
            result_summary TEXT,
            status VARCHAR(16),
            latency_ms INT,
            ts DATETIME NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_tool_calls_session (session_id),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        
        queries = create_tables_sql.strip().split(';')
        for query in queries:
            if query.strip():
                await self.execute(query.strip())
    
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
                await conn.commit()
                self._record_success()
            except Exception as e:
                await conn.rollback()
                self._record_failure()
                raise
    
    async def execute(self, query: str, params: Optional[Dict] = None) -> Any:
        start_time = time.time()
        
        try:
            async with self._get_connection() as conn:
                async with conn.cursor() as cur:
                    if params:
                        await cur.execute(query, params)
                    else:
                        await cur.execute(query)
                    
                    result = cur.lastrowid or cur.rowcount
                    
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
            async with conn.cursor() as cur:
                for query, params in queries:
                    if params:
                        await cur.execute(query, params)
                    else:
                        await cur.execute(query)
                    results.append(cur.lastrowid or cur.rowcount)
        return results
    
    async def insert(self, table: str, data: Dict) -> str:
        columns = list(data.keys())
        placeholders = ", ".join(["%s"] * len(columns))
        
        query = f"""
            INSERT INTO {table} ({', '.join(columns)})
            VALUES ({placeholders})
        """
        
        async with self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, list(data.values()))
                return str(cur.lastrowid)
    
    async def update(self, table: str, data: Dict, where: Dict) -> int:
        set_clause = ", ".join([f"{k} = %s" for k in data.keys()])
        where_clause = " AND ".join([f"{k} = %s" for k in where.keys()])
        
        query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        params = list(data.values()) + list(where.values())
        
        async with self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                return cur.rowcount
    
    async def delete(self, table: str, where: Dict) -> int:
        where_clause = " AND ".join([f"{k} = %s" for k in where.keys()])
        query = f"DELETE FROM {table} WHERE {where_clause}"
        
        async with self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, list(where.values()))
                return cur.rowcount
    
    async def select_one(
        self,
        table: str,
        columns: List[str],
        where: Dict
    ) -> Optional[Dict]:
        cols = ", ".join(columns)
        where_clause = " AND ".join([f"{k} = %s" for k in where.keys()])
        
        query = f"SELECT {cols} FROM {table} WHERE {where_clause} LIMIT 1"
        
        async with self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, list(where.values()))
                row = await cur.fetchone()
                
                if row:
                    return dict(zip(columns, row))
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
            where_clause = " AND ".join([f"{k} = %s" for k in where.keys()])
            query += f" WHERE {where_clause}"
            params = list(where.values())
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        async with self._get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                rows = await cur.fetchall()
                return [dict(zip(columns, row)) for row in rows]
    
    async def begin_transaction(self) -> Any:
        await self._init_pool()
        conn = await self._pool.acquire()
        await conn.begin()
        return conn
    
    async def commit(self, transaction: Any) -> None:
        await transaction.commit()
        await self._pool.release(transaction)
    
    async def rollback(self, transaction: Any) -> None:
        await transaction.rollback()
        await self._pool.release(transaction)
    
    async def health_check(self) -> Dict[str, Any]:
        try:
            async with self._get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
                    
                return {
                    "status": "healthy",
                    "database": self._database,
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
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None
            self._logger.info("MySQL pool closed")
