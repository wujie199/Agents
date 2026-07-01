from typing import Any, Optional

from document.rag.config import RagPipelineConfig
from document.rag.application.retrieval.rewrite.hyde import HyDERewriter
from document.rag.application.retrieval.rewrite.multi_query import MultiQueryExpander, QueryRewriterPipeline


def build_query_rewriter(
    config: RagPipelineConfig,
    llm_model: Optional[Any] = None,
) -> Optional[QueryRewriterPipeline]:
    rw = config.rewrite
    if not rw.enable_hyde and not rw.enable_multi_query:
        return None
    if llm_model is None and (rw.enable_hyde or rw.enable_multi_query):
        return None

    hyde = HyDERewriter(llm_model=llm_model) if rw.enable_hyde else None
    multi = (
        MultiQueryExpander(llm_model=llm_model, num_queries=rw.multi_query_count)
        if rw.enable_multi_query
        else None
    )
    pipeline = QueryRewriterPipeline(
        hyde_rewriter=hyde,
        multi_query_expander=multi,
        enable_hyde=rw.enable_hyde,
        enable_multi_query=rw.enable_multi_query,
    )
    return pipeline if pipeline.is_enabled() else None


def resolve_rewrite_llm(models: Any, config: RagPipelineConfig) -> Optional[Any]:
    rw = config.rewrite
    if not rw.enable_hyde and not rw.enable_multi_query:
        return None
    try:
        return models.get_model("router_llm")
    except (KeyError, ValueError, RuntimeError):
        try:
            return models.get_model("main_llm")
        except (KeyError, ValueError, RuntimeError):
            return None
