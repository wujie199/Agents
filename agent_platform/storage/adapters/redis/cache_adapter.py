import asyncio
import redis.asyncio as redis
from typing import Optional, Any, Dict
import logging
import time


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_requests: int = 3
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_requests = half_open_requests
        
        self._failures = 0
        self._last_failure_time = 0
        self._state = "closed"
        self._half_open_successes = 0
    
    def is_open(self) -> bool:
        if self._state == "open":
            if time.time() - self._last_failure_time > self._recovery_timeout:
                self._state = "half_open"
                self._half_open_successes = 0
                return False
            return True
        return False
    
    def record_success(self) -> None:
        if self._state == "half_open":
            self._half_open_successes += 1
            if self._half_open_successes >= self._half_open_requests:
                self._state = "closed"
                self._failures = 0
        elif self._state == "closed":
            self._failures = 0
    
    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure_time = time.time()
        
        if self._state == "half_open":
            self._state = "open"
        elif self._state == "closed" and self._failures >= self._failure_threshold:
            self._state = "open"
    
    @property
    def state(self) -> str:
        return self._state


class EnterpriseRedisCacheAdapter:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "agents",
        pool_size: int = 10,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_times: int = 3,
        retry_delay: float = 0.5,
        circuit_breaker_threshold: int = 5,
        enable_fallback: bool = True
    ):
        self._prefix = prefix
        self._retry_times = retry_times
        self._retry_delay = retry_delay
        self._enable_fallback = enable_fallback
        
        self._pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=pool_size,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=True
        )
        
        self._client = redis.Redis(connection_pool=self._pool)
        
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold
        )
        
        self._local_fallback: Dict[str, Any] = {}
        self._fallback_ttl: Dict[str, float] = {}
        
        self._logger = logging.getLogger(__name__)
        self._health_status = "unknown"
    
    def _make_key(self, key: str) -> str:
        if key.startswith(self._prefix):
            return key
        return f"{self._prefix}:{key}"
    
    async def _execute_with_retry(self, operation, *args, **kwargs) -> Any:
        if self._circuit_breaker.is_open():
            self._logger.warning("Circuit breaker is open, using fallback")
            raise redis.ConnectionError("Circuit breaker open")
        
        last_error = None
        
        for attempt in range(self._retry_times):
            try:
                result = await operation(*args, **kwargs)
                self._circuit_breaker.record_success()
                return result
            except (redis.ConnectionError, redis.TimeoutError) as e:
                last_error = e
                self._logger.warning(f"Redis operation failed (attempt {attempt + 1}): {e}")
                
                if attempt < self._retry_times - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
            except Exception as e:
                last_error = e
                self._logger.error(f"Redis operation error: {e}")
                break
        
        self._circuit_breaker.record_failure()
        raise last_error
    
    async def get(self, key: str) -> Optional[Any]:
        full_key = self._make_key(key)
        
        try:
            value = await self._execute_with_retry(self._client.get, full_key)
            
            if value is None:
                return None
            
            import json
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
                
        except (redis.ConnectionError, redis.TimeoutError):
            if self._enable_fallback:
                return self._local_fallback.get(full_key)
            raise
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> None:
        full_key = self._make_key(key)
        
        import json
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        else:
            value = str(value)
        
        try:
            if ttl_seconds:
                await self._execute_with_retry(
                    self._client.setex, full_key, ttl_seconds, value
                )
            else:
                await self._execute_with_retry(self._client.set, full_key, value)
            
            if self._enable_fallback:
                self._local_fallback[full_key] = value
                if ttl_seconds:
                    self._fallback_ttl[full_key] = time.time() + ttl_seconds
                    
        except (redis.ConnectionError, redis.TimeoutError):
            if self._enable_fallback:
                self._local_fallback[full_key] = value
                if ttl_seconds:
                    self._fallback_ttl[full_key] = time.time() + ttl_seconds
            else:
                raise
    
    async def delete(self, key: str) -> None:
        full_key = self._make_key(key)
        
        try:
            await self._execute_with_retry(self._client.delete, full_key)
        except (redis.ConnectionError, redis.TimeoutError):
            pass
        
        self._local_fallback.pop(full_key, None)
        self._fallback_ttl.pop(full_key, None)
    
    async def expire(self, key: str, ttl_seconds: int) -> None:
        full_key = self._make_key(key)
        
        try:
            await self._execute_with_retry(self._client.expire, full_key, ttl_seconds)
        except (redis.ConnectionError, redis.TimeoutError):
            pass
        
        self._fallback_ttl[full_key] = time.time() + ttl_seconds
    
    async def invalidate_pattern(self, pattern: str) -> int:
        full_pattern = self._make_key(pattern)
        
        try:
            keys = []
            async for key in self._client.scan_iter(match=full_pattern):
                keys.append(key)
            
            if keys:
                await self._execute_with_retry(self._client.delete, *keys)
                return len(keys)
        except (redis.ConnectionError, redis.TimeoutError):
            pass
        
        prefix = full_pattern.rstrip("*")
        keys_to_delete = [k for k in self._local_fallback.keys() if k.startswith(prefix)]
        
        for k in keys_to_delete:
            del self._local_fallback[k]
            self._fallback_ttl.pop(k, None)
        
        return len(keys_to_delete)
    
    def build_key(self, tenant_id: str, category: str, identifier: str) -> str:
        return f"{tenant_id}:{self._prefix}:{category}:{identifier}"
    
    async def exists(self, key: str) -> bool:
        full_key = self._make_key(key)
        
        try:
            result = await self._execute_with_retry(self._client.exists, full_key)
            return result > 0
        except (redis.ConnectionError, redis.TimeoutError):
            return full_key in self._local_fallback
    
    async def incr(self, key: str) -> int:
        full_key = self._make_key(key)
        
        try:
            return await self._execute_with_retry(self._client.incr, full_key)
        except (redis.ConnectionError, redis.TimeoutError):
            if full_key not in self._local_fallback:
                self._local_fallback[full_key] = 0
            self._local_fallback[full_key] = int(self._local_fallback[full_key]) + 1
            return self._local_fallback[full_key]
    
    async def health(self) -> dict:
        try:
            start = time.time()
            await self._client.ping()
            latency = (time.time() - start) * 1000
            
            info = await self._client.info()
            
            return {
                "status": "healthy",
                "type": "redis",
                "latency_ms": round(latency, 2),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "circuit_breaker_state": self._circuit_breaker.state,
                "pool_max_connections": self._pool.max_connections
            }
        except Exception as e:
            self._health_status = "unhealthy"
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_state": self._circuit_breaker.state,
                "fallback_available": self._enable_fallback
            }
    
    async def close(self) -> None:
        await self._client.close()
        await self._pool.disconnect()
        self._logger.info("Redis connection pool closed")
    
    def _cleanup_fallback(self) -> None:
        now = time.time()
        expired_keys = [k for k, v in self._fallback_ttl.items() if v < now]
        for k in expired_keys:
            self._local_fallback.pop(k, None)
            del self._fallback_ttl[k]
