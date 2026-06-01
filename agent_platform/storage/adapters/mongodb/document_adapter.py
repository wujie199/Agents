import logging
from typing import Optional, List, Any, Dict
from datetime import datetime
import time

from core.ports.storage.document import DocumentPort


class MongoDBAdapter:
    """MongoDB 文档数据库适配器
    
    企业级特性：
    - Motor 异步客户端
    - 连接池
    - 熔断器保护
    - 慢查询日志
    - 自动索引创建
    
    适用场景：
    - 对话历史（conversation_history 集合）
    - 会话存档（session_archives 集合）
    - 用户画像（user_profiles 集合）
    - 配置存储（configs 集合）
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        database: str = "agents",
        user: Optional[str] = None,
        password: Optional[str] = None,
        replica_set: Optional[str] = None,
        max_pool_size: int = 100,
        min_pool_size: int = 0,
        connect_timeout: float = 10.0,
        server_selection_timeout: float = 30.0,
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
        self._replica_set = replica_set
        self._max_pool_size = max_pool_size
        self._min_pool_size = min_pool_size
        self._connect_timeout = connect_timeout
        self._server_selection_timeout = server_selection_timeout
        self._slow_query_threshold = slow_query_threshold
        
        self._client = None
        self._db = None
        self._logger = logging.getLogger("storage.mongodb")
        
        self._enable_cb = enable_circuit_breaker
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._last_failure_time = 0
        self._circuit_open = False
    
    async def _init_client(self) -> None:
        if self._client is not None:
            return
        
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            
            if self._user and self._password:
                uri = f"mongodb://{self._user}:{self._password}@{self._host}:{self._port}/{self._database}"
            else:
                uri = f"mongodb://{self._host}:{self._port}/{self._database}"
            
            if self._replica_set:
                uri += f"?replicaSet={self._replica_set}"
            
            self._client = AsyncIOMotorClient(
                uri,
                maxPoolSize=self._max_pool_size,
                minPoolSize=self._min_pool_size,
                connectTimeoutMS=int(self._connect_timeout * 1000),
                serverSelectionTimeoutMS=int(self._server_selection_timeout * 1000),
            )
            
            self._db = self._client[self._database]
            
            self._logger.info(
                f"MongoDB client initialized: {self._host}:{self._port}/{self._database}"
            )
            
            await self._init_indexes()
            
        except Exception as e:
            self._logger.error(f"Failed to initialize MongoDB client: {e}")
            raise
    
    async def _init_indexes(self) -> None:
        indexes_to_create = {
            "conversation_history": [
                ([("session_id", 1)], False),
                ([("ts", -1)], False),
                ([("session_id", 1), ("ts", -1)], False),
            ],
            "session_archives": [
                ([("session_id", 1)], True),
                ([("user_id", 1)], False),
                ([("tenant_id", 1)], False),
                ([("started_at", -1)], False),
            ],
            "user_profiles": [
                ([("user_id", 1)], True),
                ([("tenant_id", 1)], False),
            ],
            "configs": [
                ([("key", 1)], True),
                ([("tenant_id", 1), ("key", 1)], True),
            ],
        }
        
        for collection_name, indexes in indexes_to_create.items():
            for keys, unique in indexes:
                try:
                    await self.create_index(collection_name, keys, unique)
                except Exception as e:
                    self._logger.warning(
                        f"Failed to create index on {collection_name}: {e}"
                    )
    
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
    
    async def _get_collection(self, collection: str) -> Any:
        await self._init_client()
        
        if not self._check_circuit_breaker():
            raise Exception("Circuit breaker is open")
        
        return self._db[collection]
    
    async def insert_one(
        self,
        collection: str,
        document: Dict[str, Any]
    ) -> str:
        start_time = time.time()
        
        try:
            coll = await self._get_collection(collection)
            
            if "created_at" not in document:
                document["created_at"] = datetime.utcnow()
            if "updated_at" not in document:
                document["updated_at"] = datetime.utcnow()
            
            result = await coll.insert_one(document)
            
            self._record_success()
            self._log_slow_query(start_time, "insert_one", collection)
            
            return str(result.inserted_id)
            
        except Exception as e:
            self._record_failure()
            self._logger.error(f"insert_one failed: {e}")
            raise
    
    async def insert_many(
        self,
        collection: str,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        start_time = time.time()
        
        try:
            coll = await self._get_collection(collection)
            
            now = datetime.utcnow()
            for doc in documents:
                if "created_at" not in doc:
                    doc["created_at"] = now
                if "updated_at" not in doc:
                    doc["updated_at"] = now
            
            result = await coll.insert_many(documents)
            
            self._record_success()
            self._log_slow_query(start_time, "insert_many", collection)
            
            return [str(id) for id in result.inserted_ids]
            
        except Exception as e:
            self._record_failure()
            self._logger.error(f"insert_many failed: {e}")
            raise
    
    async def find_one(
        self,
        collection: str,
        filter: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        start_time = time.time()
        
        try:
            coll = await self._get_collection(collection)
            result = await coll.find_one(filter)
            
            self._record_success()
            self._log_slow_query(start_time, "find_one", collection)
            
            return result
            
        except Exception as e:
            self._record_failure()
            self._logger.error(f"find_one failed: {e}")
            raise
    
    async def find_many(
        self,
        collection: str,
        filter: Dict[str, Any],
        sort: Optional[List[tuple]] = None,
        skip: int = 0,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        start_time = time.time()
        
        try:
            coll = await self._get_collection(collection)
            
            cursor = coll.find(filter)
            
            if sort:
                cursor = cursor.sort(sort)
            if skip > 0:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            
            results = await cursor.to_list(length=None if limit is None else limit)
            
            self._record_success()
            self._log_slow_query(start_time, "find_many", collection)
            
            return results
            
        except Exception as e:
            self._record_failure()
            self._logger.error(f"find_many failed: {e}")
            raise
    
    async def update_one(
        self,
        collection: str,
        filter: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False
    ) -> bool:
        start_time = time.time()
        
        try:
            coll = await self._get_collection(collection)
            
            if "$set" in update:
                update["$set"]["updated_at"] = datetime.utcnow()
            else:
                update = {"$set": {**update, "updated_at": datetime.utcnow()}}
            
            result = await coll.update_one(filter, update, upsert=upsert)
            
            self._record_success()
            self._log_slow_query(start_time, "update_one", collection)
            
            return result.modified_count > 0 or (upsert and result.upserted_id is not None)
            
        except Exception as e:
            self._record_failure()
            self._logger.error(f"update_one failed: {e}")
            raise
    
    async def update_many(
        self,
        collection: str,
        filter: Dict[str, Any],
        update: Dict[str, Any]
    ) -> int:
        start_time = time.time()
        
        try:
            coll = await self._get_collection(collection)
            
            if "$set" in update:
                update["$set"]["updated_at"] = datetime.utcnow()
            else:
                update = {"$set": {**update, "updated_at": datetime.utcnow()}}
            
            result = await coll.update_many(filter, update)
            
            self._record_success()
            self._log_slow_query(start_time, "update_many", collection)
            
            return result.modified_count
            
        except Exception as e:
            self._record_failure()
            self._logger.error(f"update_many failed: {e}")
            raise
    
    async def delete_one(
        self,
        collection: str,
        filter: Dict[str, Any]
    ) -> bool:
        start_time = time.time()
        
        try:
            coll = await self._get_collection(collection)
            result = await coll.delete_one(filter)
            
            self._record_success()
            self._log_slow_query(start_time, "delete_one", collection)
            
            return result.deleted_count > 0
            
        except Exception as e:
            self._record_failure()
            self._logger.error(f"delete_one failed: {e}")
            raise
    
    async def delete_many(
        self,
        collection: str,
        filter: Dict[str, Any]
    ) -> int:
        start_time = time.time()
        
        try:
            coll = await self._get_collection(collection)
            result = await coll.delete_many(filter)
            
            self._record_success()
            self._log_slow_query(start_time, "delete_many", collection)
            
            return result.deleted_count
            
        except Exception as e:
            self._record_failure()
            self._logger.error(f"delete_many failed: {e}")
            raise
    
    async def count(
        self,
        collection: str,
        filter: Dict[str, Any]
    ) -> int:
        start_time = time.time()
        
        try:
            coll = await self._get_collection(collection)
            result = await coll.count_documents(filter)
            
            self._record_success()
            self._log_slow_query(start_time, "count", collection)
            
            return result
            
        except Exception as e:
            self._record_failure()
            self._logger.error(f"count failed: {e}")
            raise
    
    async def aggregate(
        self,
        collection: str,
        pipeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        start_time = time.time()
        
        try:
            coll = await self._get_collection(collection)
            cursor = coll.aggregate(pipeline)
            results = await cursor.to_list(length=None)
            
            self._record_success()
            self._log_slow_query(start_time, "aggregate", collection)
            
            return results
            
        except Exception as e:
            self._record_failure()
            self._logger.error(f"aggregate failed: {e}")
            raise
    
    async def create_index(
        self,
        collection: str,
        keys: List[tuple],
        unique: bool = False
    ) -> str:
        await self._init_client()
        
        coll = self._db[collection]
        
        index_name = await coll.create_index(keys, unique=unique)
        
        self._logger.info(f"Created index {index_name} on {collection}")
        
        return index_name
    
    def _log_slow_query(
        self,
        start_time: float,
        operation: str,
        collection: str
    ) -> None:
        elapsed = time.time() - start_time
        
        if elapsed > self._slow_query_threshold:
            self._logger.warning(
                f"Slow query ({elapsed:.3f}s): {operation} on {collection}"
            )
    
    async def health_check(self) -> Dict[str, Any]:
        try:
            await self._init_client()
            
            result = await self._db.command("ping")
            
            server_info = await self._db.command("serverStatus")
            
            return {
                "status": "healthy",
                "database": self._database,
                "connections": server_info.get("connections", {}),
                "circuit_breaker": "closed" if not self._circuit_open else "open",
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker": "closed" if not self._circuit_open else "open",
            }
    
    async def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._logger.info("MongoDB client closed")
