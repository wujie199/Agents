"""生产/开发环境 RunContext 装配入口。

组合 infrastructure_factory、storage_factory、memory_helpers、rag_factory_helpers
等辅助工厂，构建完整的 RunContext。
"""

import os
from pathlib import Path
from typing import Any, Optional, Tuple

from core.domain.context import RequestContext
from core.composition.run_context import RunContext
from core.composition.infrastructure_factory import (
    build_infrastructure_ports,
)
from core.composition.storage_factory import (
    build_cache_port,
    build_storage_ports,
)
from core.composition.rag_factory_helpers import build_rag_stack
from core.composition.memory_helpers import (
    build_checkpointer,
    build_turn_buffer,
)

from agent_platform.infrastructure.skills.adapter import SimpleSkillAdapter
from agent_platform.infrastructure.mcp.adapter import EnterpriseMCPAdapter
from agent_platform.memory.adapters.config_loader import load_memory_config


def _resolve_dev_rag_paths(data_dir: str) -> Tuple[str, str]:
    """开发环境 RAG 路径：优先使用离线建库目录 data/rag_offline。"""
    base = Path(data_dir)
    offline = base / "rag_offline"
    offline_chroma = offline / "chroma_dev"
    if offline_chroma.is_dir() and any(offline_chroma.iterdir()):
        return str(offline_chroma), str(offline)
    return str(base / "chroma_dev"), str(base)


# ── Memory Port 构建 ──

def _build_memory_port(
    config_dir: str,
    data_dir: str,
    archive_db: Any,
    privacy: Any,
    skills: SimpleSkillAdapter,
    cache: Any = None,
    models: Any = None,
    store_dir_override: Optional[str] = None,
    session_vector_index: Any = None,
    session_hybrid_search: bool = True,
    object_store: Any = None,
    index_port: Any = None,
    secret: Any = None,
    mem_cfg_override: Optional[dict] = None,
) -> Any:
    from agent_platform.memory.adapters.memory_port_adapter import MemoryPortAdapter
    from agent_platform.memory.adapters.hot_memory_compressor_adapter import (
        LlmHotMemoryCompressorAdapter,
        TruncatingHotMemoryCompressorAdapter,
    )
    from agent_platform.memory.adapters.llm_summarizer_adapter import (
        LlmMemorySummarizerAdapter,
    )
    from agent_platform.memory.adapters.skill_memory_adapter import SkillMemoryAdapter
    from agent_platform.memory.adapters.summarizer_adapter import TruncatingSummarizerAdapter
    from agent_platform.memory.adapters.session_vector_factory import build_session_vector_index
    from agent_platform.memory.adapters.external_factory import build_external_memory
    from agent_platform.memory.adapters.field_crypto import resolve_encryption_key
    from core.composition.memory_helpers import build_hot_memory

    cfg = load_memory_config(f"{config_dir}/memory.yml")
    if mem_cfg_override:
        cfg = {**cfg, **mem_cfg_override}
    hot = build_hot_memory(
        cfg,
        archive_db=archive_db,
        store_dir_override=store_dir_override,
    )
    fallback_summarizer = TruncatingSummarizerAdapter(
        max_chars=cfg.get("session_search_max_chars", 2000)
    )
    summarizer_role = cfg.get("memory_summarizer_role", "memory_summarizer_llm")
    if cfg.get("use_llm_summarize", True) and models is not None:
        summarizer = LlmMemorySummarizerAdapter(
            models=models,
            role=summarizer_role,
            max_chars=cfg.get("session_search_max_chars", 2000),
            fallback=fallback_summarizer,
        )
    else:
        summarizer = fallback_summarizer

    if cfg.get("use_llm_compress", True) and models is not None:
        compressor = LlmHotMemoryCompressorAdapter(
            models=models,
            role=summarizer_role,
        )
    else:
        compressor = TruncatingHotMemoryCompressorAdapter()

    skill_memory = SkillMemoryAdapter(
        skills=skills,
        drafts_dir=cfg.get("skills_drafts_dir", "skills/drafts"),
        meta_dir=cfg.get("skills_meta_dir", "skills/meta"),
        published_dir=cfg.get("skills_dir", "skills/published"),
        archive_db=archive_db,
        auto_extract_draft=cfg.get("skill_auto_extract_draft", False),
        deprecate_threshold=float(cfg.get("skill_deprecate_threshold", 0.2)),
        include_deprecated_in_search=cfg.get(
            "skill_include_deprecated_in_search", False
        ),
        auto_extract_min_steps=int(cfg.get("skill_auto_extract_min_steps", 2)),
    )
    external = build_external_memory(cfg)
    session_vector_index = build_session_vector_index(
        cfg, data_dir=data_dir, config_dir=config_dir, models=models
    )
    encrypt_at_rest = cfg.get("cold_archive_encrypt_at_rest", False)
    encryption_key = resolve_encryption_key(
        cfg.get("memory_encryption_key"),
        secret_port=secret,
    )

    resolved_store = store_dir_override or cfg.get("store_dir") or f"{data_dir}/memory_dev"

    return MemoryPortAdapter(
        store_dir=str(resolved_store),
        archive_db=archive_db,
        hot_memory=hot,
        privacy=privacy,
        skill_memory=skill_memory,
        summarizer=summarizer,
        compressor=compressor,
        external_memory=external,
        cache=cache,
        models=models,
        hot_memory_max_chars=cfg.get("hot_memory_max_chars", 2200),
        user_memory_max_chars=cfg.get("user_memory_max_chars", 1375),
        session_search_cache_ttl=cfg.get("session_search_cache_ttl", 900),
        session_search_negative_cache_ttl=cfg.get(
            "session_search_negative_cache_ttl", 120
        ),
        retention_days=cfg.get("retention_days", 90),
        session_vector_index=session_vector_index,
        session_hybrid_search=cfg.get("session_hybrid_search", True),
        object_store=object_store,
        enable_cold_archive=cfg.get("enable_cold_archive", False),
        cold_archive_prefix=cfg.get("cold_archive_prefix", "l2/cold"),
        cold_archive_compress=cfg.get("cold_archive_compress", True),
        session_search_cold_fallback=cfg.get("session_search_cold_fallback", True),
        session_search_rerank=cfg.get("session_search_rerank", True),
        cold_archive_search_scan_limit=cfg.get("cold_archive_search_scan_limit", 100),
        cold_archive_keep_vectors=cfg.get("cold_archive_keep_vectors", True),
        session_vector_auto_reindex=cfg.get(
            "session_vector_auto_reindex_on_version_change", True
        ),
        reindex_batch_size=cfg.get("reindex_batch_size", 200),
        index_port=index_port,
        cold_archive_encrypt_at_rest=encrypt_at_rest,
        encryption_key=encryption_key,
        external_merge_on_finalize=cfg.get("external_merge_on_finalize", True),
        purge_delete_external_audit=cfg.get("purge_delete_external_audit", True),
        purge_tenant_l4_strip_user_keys=cfg.get(
            "purge_tenant_l4_strip_user_keys", True
        ),
    )


# ── 核心装配逻辑 ──

def _assemble_run_context(
    request: RequestContext,
    config_dir: str,
    data_dir: str,
    models: Any,
    tools: Any,
    skills: SimpleSkillAdapter,
    mcp: Any,
    infra: Any,
    storage: Any,
    rag: Any,
    index_port: Any,
    knowledge_base: Any,
    memory: Any,
    mem_cfg: dict,
    extra: dict,
) -> RunContext:
    """将已构建的各端口组装为 RunContext。"""
    turn_buffer = build_turn_buffer(memory, mem_cfg)
    checkpointer = build_checkpointer(storage.relational)

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
        policy=infra.policy,
        privacy=infra.privacy,
        observability=infra.observability,
        identity=infra.identity,
        turn_buffer=turn_buffer,
        checkpointer=checkpointer,
        extra=extra,
    )


# ── 公共入口 ──

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
    use_memory_graph: bool = True,
    **overrides,
) -> RunContext:
    """构建生产环境 RunContext（企业级实现）。"""
    # ── L1 基础设施 ──
    infra = build_infrastructure_ports(config_dir=config_dir, service_name="agents")

    # ── 记忆配置 ──
    mem_cfg = load_memory_config(f"{config_dir}/memory.yml")

    # ── L3 存储 ──
    storage = build_storage_ports(
        mem_cfg=mem_cfg,
        data_dir=data_dir,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_password,
        cache_pool_size=20,
        use_memory_graph=use_memory_graph,
        s3_endpoint=s3_endpoint,
        s3_access_key=s3_access_key,
        s3_secret_key=s3_secret_key,
        s3_bucket=s3_bucket,
    )

    # ── Skills & MCP ──
    skills = SimpleSkillAdapter(skills_dir="skills/published")
    mcp = EnterpriseMCPAdapter(
        config_path=f"{config_dir}/mcp_servers.yml",
        max_connections_per_server=3,
        default_timeout=30.0,
        health_check_interval=30.0,
        circuit_breaker_threshold=5,
    )

    # ── 模型 ──
    from agent_platform.model.registry import ModelRegistry
    models = ModelRegistry(config_path=f"{config_dir}/models.yml")

    # ── RAG 栈 ──
    stack = build_rag_stack(
        models=models,
        vector_port=storage.vector,
        cache_port=storage.cache,
        config_dir=config_dir,
        sql_port=storage.relational,
        graph_port=storage.graph,
        privacy_port=infra.privacy,
        data_dir=data_dir,
    )
    rag = stack.rag
    index_port = stack.index_port
    knowledge_base = stack.knowledge_base

    # ── 工具 ──
    from agent_platform.tools.adapters.tool_port_adapter import ToolPortAdapter
    tools = ToolPortAdapter(config_path=f"{config_dir}/tools.yml")

    # ── 记忆 ──
    memory = _build_memory_port(
        config_dir=config_dir,
        data_dir=data_dir,
        archive_db=storage.relational,
        privacy=infra.privacy,
        skills=skills,
        cache=storage.cache,
        models=models,
        store_dir_override=None,
        object_store=storage.object_store,
        index_port=index_port,
        secret=infra.secret,
    )

    from agent_platform.memory.memory_tool_registration import register_memory_tools
    register_memory_tools(tools, memory)

    return _assemble_run_context(
        request=request,
        config_dir=config_dir,
        data_dir=data_dir,
        models=models,
        tools=tools,
        skills=skills,
        mcp=mcp,
        infra=infra,
        storage=storage,
        rag=rag,
        index_port=index_port,
        knowledge_base=knowledge_base,
        memory=memory,
        mem_cfg=mem_cfg,
        extra={
            "config": infra.config,
            "secret": infra.secret,
            "cache": storage.cache,
            "relational": storage.relational,
            "graph": storage.graph,
            "object_store": storage.object_store,
            "vector_port": storage.vector,
            "data_dir": data_dir,
            "memory_config_summary": {
                "memory_config_path": mem_cfg.get("_config_path"),
                "archive_backend": mem_cfg.get("archive_backend", "sqlite"),
                "l1_store_backend": mem_cfg.get("l1_store_backend", "file"),
                "store_dir": mem_cfg.get("store_dir"),
                "enable_cold_archive": mem_cfg.get("enable_cold_archive"),
                "enable_session_vector_index": mem_cfg.get(
                    "enable_session_vector_index"
                ),
            },
        },
    )


def build_development_context(
    request: RequestContext,
    config_dir: str = "config",
    data_dir: str = "data",
    **overrides,
) -> RunContext:
    """构建开发环境 RunContext。"""
    # ── L1 基础设施 ──
    infra = build_infrastructure_ports(config_dir=config_dir, service_name="agents-dev")

    # ── 记忆配置 ──
    mem_cfg = load_memory_config(f"{config_dir}/memory.yml")

    # ── L3 存储 ──
    storage = build_storage_ports(
        mem_cfg=mem_cfg,
        data_dir=data_dir,
        relational_db_name="dev_archive.db",
        cache_pool_size=10,
        use_memory_graph=True,
    )

    # ── Skills & MCP ──
    skills = SimpleSkillAdapter(skills_dir="skills/published")
    mcp = EnterpriseMCPAdapter(config_path=f"{config_dir}/mcp_servers.yml")

    # ── 模型 ──
    from agent_platform.model.registry import ModelRegistry
    models = ModelRegistry(config_path=f"{config_dir}/models.yml")

    # ── RAG 栈（开发路径） ──
    chroma_dir, rag_data_dir = _resolve_dev_rag_paths(data_dir)
    from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter
    dev_vector = ChromaVectorAdapter(persist_directory=chroma_dir)
    stack = build_rag_stack(
        models=models,
        vector_port=dev_vector,
        cache_port=storage.cache,
        config_dir=config_dir,
        sql_port=storage.relational,
        graph_port=storage.graph,
        privacy_port=infra.privacy,
        data_dir=rag_data_dir,
    )
    rag = stack.rag
    index_port = stack.index_port
    knowledge_base = stack.knowledge_base

    # ── 工具 ──
    from agent_platform.tools.adapters.tool_port_adapter import ToolPortAdapter
    tools = ToolPortAdapter(config_path=f"{config_dir}/tools.yml")

    # ── 记忆 ──
    memory = _build_memory_port(
        config_dir=config_dir,
        data_dir=data_dir,
        archive_db=storage.relational,
        privacy=infra.privacy,
        skills=skills,
        cache=storage.cache,
        models=models,
        store_dir_override=f"{data_dir}/memory_dev",
        object_store=storage.object_store,
        index_port=index_port,
        secret=infra.secret,
    )

    from agent_platform.memory.memory_tool_registration import register_memory_tools
    register_memory_tools(tools, memory)

    return _assemble_run_context(
        request=request,
        config_dir=config_dir,
        data_dir=data_dir,
        models=models,
        tools=tools,
        skills=skills,
        mcp=mcp,
        infra=infra,
        storage=storage,
        rag=rag,
        index_port=index_port,
        knowledge_base=knowledge_base,
        memory=memory,
        mem_cfg=mem_cfg,
        extra={
            "config": infra.config,
            "secret": infra.secret,
            "cache": storage.cache,
            "relational": storage.relational,
            "graph": storage.graph,
            "object_store": storage.object_store,
            "vector_port": dev_vector,
            "rag_chroma_dir": chroma_dir,
            "rag_tenant_id": None,
            "data_dir": data_dir,
        },
    )
