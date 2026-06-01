import os
from typing import Optional
from core.domain.context import RequestContext
from agent_platform.infrastructure.config.adapter import ConfigPortAdapter
from agent_platform.infrastructure.secret.adapter import SecretPortAdapter
from agent_platform.infrastructure.privacy.adapter import PrivacyPortAdapter
from agent_platform.infrastructure.identity.adapter import IdentityPortAdapter
from agent_platform.infrastructure.policy.adapter import PolicyPortAdapter
from agent_platform.infrastructure.observability.adapter import ObservabilityPortAdapter
from agent_platform.infrastructure.skills.adapter import SimpleSkillAdapter
from agent_platform.infrastructure.mcp.adapter import EnterpriseMCPAdapter
from agent_platform.storage.adapters.redis.cache_adapter import EnterpriseRedisCacheAdapter
from agent_platform.storage.adapters.sqlite.relational_adapter import AsyncSQLiteRelationalAdapter
from agent_platform.storage.adapters.graph.memory_graph_adapter import MemoryGraphAdapter
from agent_platform.storage.adapters.s3.s3_object_store_adapter import S3ObjectStoreAdapter
from core.composition.run_context import RunContext
from core.composition.rag_factory_helpers import build_rag_stack
from agent_platform.storage.adapters.memory.async_cache_adapter import AsyncMemoryCacheAdapter


def build_production_context(
    request: RequestContext,
    config_dir: str = "config",
    data_dir: str = "data",
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_password: Optional[str] = None,
    s3_endpoint: Optional[str] = None,
    s3_access_key: Optional[str] = None,
    s3_secret_key: Optional[str] = None,
    s3_bucket: str = "agents-storage",
    use_memory_cache: bool = False,
    use_memory_graph: bool = True,
    **overrides
) -> RunContext:
    """
    构建生产环境 RunContext（企业级实现）。

    Args:
        request: RequestContext
        config_dir: 配置文件目录
        data_dir: 数据目录
        redis_host: Redis 主机
        redis_port: Redis 端口
        redis_password: Redis 密码
        s3_endpoint: S3/OBS 端点
        s3_access_key: S3 Access Key
        s3_secret_key: S3 Secret Key
        s3_bucket: S3 桶名
        use_memory_cache: 使用内存缓存（开发用）
        use_memory_graph: 使用内存图库（开发用）
        **overrides: 覆盖指定 Port
    """
    config = ConfigPortAdapter(config_dir=config_dir)
    secret = SecretPortAdapter()
    privacy = PrivacyPortAdapter()
    identity = IdentityPortAdapter()
    policy = PolicyPortAdapter(config_path=f"{config_dir}/concurrency.yml")
    observability = ObservabilityPortAdapter(service_name="agents")
    
    if use_memory_cache:
        cache = AsyncMemoryCacheAdapter()
    else:
        cache = EnterpriseRedisCacheAdapter(
            host=redis_host,
            port=redis_port,
            password=redis_password or os.environ.get("REDIS_PASSWORD"),
            pool_size=20,
            retry_times=3,
            circuit_breaker_threshold=5,
            enable_fallback=True
        )
    
    relational = AsyncSQLiteRelationalAdapter(
        db_path=f"{data_dir}/session_archive.db",
        pool_size=10,
        timeout=30.0
    )
    
    if use_memory_graph:
        graph = MemoryGraphAdapter()
    else:
        from agent_platform.storage.adapters.neo4j.neo4j_graph_adapter import Neo4jGraphAdapter
        graph = Neo4jGraphAdapter(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", ""),
            max_connection_pool_size=50
        )
    
    object_store = S3ObjectStoreAdapter(
        endpoint_url=s3_endpoint or os.environ.get("S3_ENDPOINT"),
        access_key=s3_access_key or os.environ.get("S3_ACCESS_KEY", ""),
        secret_key=s3_secret_key or os.environ.get("S3_SECRET_KEY", ""),
        bucket_name=os.environ.get("S3_BUCKET", s3_bucket)
    )
    
    skills = SimpleSkillAdapter(skills_dir="skills/published")
    
    mcp = EnterpriseMCPAdapter(
        config_path=f"{config_dir}/mcp_servers.yml",
        max_connections_per_server=3,
        default_timeout=30.0,
        health_check_interval=30.0,
        circuit_breaker_threshold=5
    )
    
    from agent_platform.model.registry import ModelRegistry
    models = ModelRegistry(config_path=f"{config_dir}/models.yml")
    
    from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter

    vector_port = ChromaVectorAdapter(persist_directory=f"{data_dir}/chroma")
    rag, index_port, knowledge_base, _, _ = build_rag_stack(
        models=models,
        vector_port=vector_port,
        cache_port=cache,
        config_dir=config_dir,
        sql_port=relational,
        graph_port=graph,
        privacy_port=privacy,
    )

    from agent_platform.tools.adapters.tool_port_adapter import ToolPortAdapter
    tools = ToolPortAdapter(config_path=f"{config_dir}/tools.yml")
    
    from agent_platform.memory.adapters.memory_port_adapter import MemoryPortAdapter
    memory = MemoryPortAdapter(
        store_dir=f"{data_dir}/memory",
        archive_db=relational
    )
    
    return RunContext(
        request=request,
        rag=rag,
        index=index_port,
        knowledge_base=knowledge_base,
        memory=memory,
        tools=tools,
        skills=skills,
        mcp=mcp,
        models=models,
        policy=policy,
        privacy=privacy,
        observability=observability,
        identity=identity,
        extra={
            "config": config,
            "secret": secret,
            "cache": cache,
            "relational": relational,
            "graph": graph,
            "object_store": object_store,
            "vector_port": vector_port,
        }
    )


def build_development_context(
    request: RequestContext,
    config_dir: str = "config",
    data_dir: str = "data",
    **overrides
) -> RunContext:
    """
    构建开发环境 RunContext。

    Args:
        request: RequestContext
        config_dir: 配置文件目录
        data_dir: 数据目录
        **overrides: 覆盖指定 Port
    """
    config = ConfigPortAdapter(config_dir=config_dir)
    secret = SecretPortAdapter()
    privacy = PrivacyPortAdapter()
    identity = IdentityPortAdapter()
    policy = PolicyPortAdapter(config_path=f"{config_dir}/concurrency.yml")
    observability = ObservabilityPortAdapter(service_name="agents-dev")
    
    cache = AsyncMemoryCacheAdapter()
    
    relational = AsyncSQLiteRelationalAdapter(
        db_path=f"{data_dir}/dev_archive.db",
        pool_size=3,
        timeout=10.0
    )
    
    graph = MemoryGraphAdapter()
    
    object_store = S3ObjectStoreAdapter()
    
    skills = SimpleSkillAdapter(skills_dir="skills/published")
    
    mcp = EnterpriseMCPAdapter(config_path=f"{config_dir}/mcp_servers.yml")
    
    from agent_platform.model.registry import ModelRegistry
    models = ModelRegistry(config_path=f"{config_dir}/models.yml")
    
    from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter

    vector_port = ChromaVectorAdapter(persist_directory=f"{data_dir}/chroma_dev")
    rag, index_port, knowledge_base, _, _ = build_rag_stack(
        models=models,
        vector_port=vector_port,
        cache_port=cache,
        config_dir=config_dir,
        sql_port=relational,
        graph_port=graph,
        privacy_port=privacy,
    )
    rag._enable_cache = False
    
    from agent_platform.tools.adapters.tool_port_adapter import ToolPortAdapter
    tools = ToolPortAdapter(config_path=f"{config_dir}/tools.yml")
    
    from agent_platform.memory.adapters.memory_port_adapter import MemoryPortAdapter
    memory = MemoryPortAdapter(
        store_dir=f"{data_dir}/memory_dev",
        archive_db=relational
    )
    
    return RunContext(
        request=request,
        rag=rag,
        index=index_port,
        knowledge_base=knowledge_base,
        memory=memory,
        tools=tools,
        skills=skills,
        mcp=mcp,
        models=models,
        policy=policy,
        privacy=privacy,
        observability=observability,
        identity=identity,
        extra={
            "config": config,
            "secret": secret,
            "cache": cache,
            "relational": relational,
            "graph": graph,
            "object_store": object_store,
            "vector_port": vector_port,
        }
    )
