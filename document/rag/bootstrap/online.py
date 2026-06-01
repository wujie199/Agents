"""在线 RAG 组装：检索、建库门面、embedding/rerank 注入。"""

from pathlib import Path
from typing import Any, Optional, Tuple

from core.ports.index import IndexProfile

from document.rag.adapters.registry import build_bm25_index, build_embedding, build_rerank
from document.rag.application.ingest.factory import build_ingest_pipeline
from document.rag.application.retrieval.rewrite_factory import (
    build_query_rewriter,
    resolve_rewrite_llm,
)
from document.rag.config import RagPipelineConfig, load_rag_pipeline_config
from document.rag.facades.knowledge_base import KnowledgeBasePortAdapter
from document.rag.facades.rag import RAGPortAdapter
from document.rag.application.cleaning.pipeline import parse_cleaning_level
from document.rag.application.indexing.service import IndexService


def resolve_embedding_model(
    models: Any,
    config: Optional[RagPipelineConfig] = None,
    *,
    config_dir: str = "config",
) -> Any:
    _ = models
    return build_embedding(config, config_dir=config_dir)


def resolve_rerank_model(models: Any, config: RagPipelineConfig) -> Any:
    _ = models
    return build_rerank(config)


def build_retrieval_router(
    vector_port: Any,
    cache_port: Any,
    embedding_model: Any,
    config: RagPipelineConfig,
    rerank_model: Any = None,
    sql_port: Any = None,
    graph_port: Any = None,
    query_rewriter: Any = None,
) -> Any:
    from document.rag.application.retrieval.router.router import RetrievalRouter

    return RetrievalRouter(
        vector_port=vector_port,
        cache_port=cache_port,
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        sql_port=sql_port,
        graph_port=graph_port,
        query_rewriter=query_rewriter,
        collection_name=config.collection_name,
        enable_cache=config.enable_cache,
        cache_ttl=config.cache_ttl_seconds,
        enable_graph=config.retrieval.enable_graph,
        enable_sql=config.retrieval.enable_sql,
        enable_rerank=config.retrieval.enable_rerank,
        default_top_k=config.default_top_k,
        default_rerank_n=config.rerank_top_n,
    )


def _default_index_profile(config: RagPipelineConfig) -> IndexProfile:
    if config.enable_graph_index and config.retrieval.enable_sql:
        return IndexProfile.FULL
    if config.retrieval.enable_sql:
        return IndexProfile.SQL_SIDECAR
    if config.enable_graph_index:
        return IndexProfile.GRAPH_SIDECAR
    return IndexProfile.VECTOR_ONLY


def build_rag_stack(
    models: Any,
    vector_port: Any,
    cache_port: Any,
    config_dir: str = "config",
    sql_port: Optional[Any] = None,
    graph_port: Optional[Any] = None,
    privacy_port: Optional[Any] = None,
    data_dir: Optional[str] = None,
) -> Tuple[Any, Any, Any, Any, RagPipelineConfig]:
    """
    Returns:
        rag, index_port, knowledge_base, embedding_model, config
    """
    rag_config = load_rag_pipeline_config(config_dir=config_dir)
    embedding = resolve_embedding_model(models, rag_config, config_dir=config_dir)
    rerank = resolve_rerank_model(models, rag_config)
    rewrite_llm = resolve_rewrite_llm(models, rag_config)
    query_rewriter = build_query_rewriter(rag_config, rewrite_llm)

    index_sql = sql_port if rag_config.retrieval.enable_sql else None
    index_graph = graph_port if rag_config.enable_graph_index else None

    bm25_index = None
    if data_dir and rag_config.retrieval.enable_bm25_search:
        bm25_index = build_bm25_index(Path(data_dir), rag_config)

    index_port = IndexService(
        vector_port=vector_port,
        embedding_model=embedding,
        config=rag_config,
        cache_port=cache_port,
        sql_port=index_sql,
        graph_port=index_graph,
        bm25_index=bm25_index,
    )

    router = None
    if rag_config.retrieval.enable_router:
        router = build_retrieval_router(
            vector_port=vector_port,
            cache_port=cache_port,
            embedding_model=embedding,
            config=rag_config,
            rerank_model=rerank,
            sql_port=sql_port if rag_config.retrieval.enable_sql else None,
            graph_port=graph_port if rag_config.retrieval.enable_graph else None,
            query_rewriter=query_rewriter,
        )

    rag = RAGPortAdapter(
        vector_port=vector_port,
        cache_port=cache_port,
        embedding_model=embedding,
        rerank_model=rerank,
        config=rag_config,
        router=router,
    )

    ingest_port = build_ingest_pipeline(rag_config)
    knowledge_base = KnowledgeBasePortAdapter(
        ingest_port=ingest_port,
        index_port=index_port,
        privacy_port=privacy_port,
        default_index_profile=_default_index_profile(rag_config),
        cleaning_level=parse_cleaning_level(rag_config.ingest.cleaning_level),
        rag_config=rag_config,
    )

    return rag, index_port, knowledge_base, embedding, rag_config
