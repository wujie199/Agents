"""L3 存储 Port 装配工厂。

构建 CachePort、RelationalPort、GraphPort、ObjectStorePort、VectorPort。
"""

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from urllib.parse import urlparse

from agent_platform.storage.adapters.redis.cache_adapter import EnterpriseRedisCacheAdapter
from agent_platform.storage.adapters.graph.memory_graph_adapter import MemoryGraphAdapter
from agent_platform.storage.adapters.s3.s3_object_store_adapter import S3ObjectStoreAdapter
from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter


def _parse_redis_url(url: str) -> Tuple[str, int, Optional[str], int]:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    db = 0
    if parsed.path and parsed.path != "/":
        try:
            db = int(parsed.path.lstrip("/"))
        except ValueError:
            db = 0
    return host, port, parsed.password, db


def build_cache_port(
    *,
    redis_host: Optional[str] = None,
    redis_port: Optional[int] = None,
    redis_password: Optional[str] = None,
    redis_db: int = 0,
    pool_size: int = 10,
    prefix: Optional[str] = None,
) -> EnterpriseRedisCacheAdapter:
    """构建 CachePort（Redis；支持 REDIS_URL / REDIS_HOST）。"""
    if prefix is None:
        prefix = os.environ.get("CHAT_CACHE_REDIS_PREFIX", "agents")
    host = redis_host or os.environ.get("REDIS_HOST", "localhost")
    port = redis_port if redis_port is not None else int(
        os.environ.get("REDIS_PORT", "6379")
    )
    password = (
        redis_password
        if redis_password is not None
        else os.environ.get("REDIS_PASSWORD")
    )
    db = redis_db
    redis_url = os.environ.get("REDIS_URL") or os.environ.get(
        "CHAT_RATE_LIMIT_REDIS_URL"
    )
    if redis_url:
        url_host, url_port, url_password, url_db = _parse_redis_url(redis_url)
        if redis_host is None:
            host = url_host
        if redis_port is None:
            port = url_port
        if password is None:
            password = url_password
        if redis_db == 0:
            db = url_db

    return EnterpriseRedisCacheAdapter(
        host=host,
        port=port,
        db=db,
        password=password,
        prefix=prefix,
        pool_size=pool_size,
        retry_times=3,
        circuit_breaker_threshold=5,
        enable_fallback=True,
    )


@dataclass
class StoragePorts:
    """L3 存储端口集合。"""
    cache: Any
    relational: Any
    graph: Any
    object_store: Any
    vector: Any


def build_storage_ports(
    *,
    mem_cfg: dict,
    data_dir: str = "data",
    relational_db_name: Optional[str] = None,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_password: Optional[str] = None,
    cache_pool_size: int = 10,
    use_memory_graph: bool = True,
    s3_endpoint: Optional[str] = None,
    s3_access_key: Optional[str] = None,
    s3_secret_key: Optional[str] = None,
    s3_bucket: str = "agents-storage",
    vector_persist_dir: Optional[str] = None,
) -> StoragePorts:
    """构建所有 L3 存储端口。

    Args:
        mem_cfg: 记忆配置字典
        data_dir: 数据根目录
        relational_db_name: 关系数据库文件名（None 则使用默认名）
        redis_host: Redis 主机
        redis_port: Redis 端口
        redis_password: Redis 密码
        cache_pool_size: Redis 连接池大小
        use_memory_graph: 使用内存图库（开发用），False 则用 Neo4j
        s3_endpoint: S3/OBS 端点
        s3_access_key: S3 Access Key
        s3_secret_key: S3 Secret Key
        s3_bucket: S3 桶名
        vector_persist_dir: Chroma 持久化目录
    """
    from agent_platform.memory.adapters.archive_factory import build_archive_db

    cache = build_cache_port(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_password,
        pool_size=cache_pool_size,
    )

    relational = build_archive_db(
        mem_cfg,
        data_dir=data_dir,
        db_name=relational_db_name,
    )

    if use_memory_graph:
        graph = MemoryGraphAdapter()
    else:
        from agent_platform.storage.adapters.neo4j.neo4j_graph_adapter import Neo4jGraphAdapter
        graph = Neo4jGraphAdapter(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", ""),
            max_connection_pool_size=50,
        )

    object_store = S3ObjectStoreAdapter(
        endpoint_url=s3_endpoint or os.environ.get("S3_ENDPOINT"),
        access_key=s3_access_key or os.environ.get("S3_ACCESS_KEY", ""),
        secret_key=s3_secret_key or os.environ.get("S3_SECRET_KEY", ""),
        bucket_name=os.environ.get("S3_BUCKET", s3_bucket),
    )

    persist_dir = vector_persist_dir or f"{data_dir}/chroma"
    vector = ChromaVectorAdapter(persist_directory=persist_dir)

    return StoragePorts(
        cache=cache,
        relational=relational,
        graph=graph,
        object_store=object_store,
        vector=vector,
    )
