from typing import Any, Optional

from document.rag.config import RagPipelineConfig
from document.rag.application.retrieval.rewrite.hyde import HyDERewriter
from document.rag.application.retrieval.rewrite.multi_query import (
    MultiQueryExpander,
    QueryRewriterPipeline,
)
from document.rag.application.retrieval.rewrite.rule_based import RuleBasedQueryRewriter
from document.rag.application.retrieval.rewrite.combined import CombinedQueryRewriter


def build_query_rewriter(
    config: RagPipelineConfig,
    llm_model: Optional[Any] = None,
) -> Optional[CombinedQueryRewriter]:
    rw = config.rewrite
    rule = (
        RuleBasedQueryRewriter(max_queries=rw.rule_max_queries)
        if rw.enable_rule_rewrite
        else None
    )

    llm_pipeline: Optional[QueryRewriterPipeline] = None
    if rw.needs_llm_capability() and llm_model is not None:
        hyde = HyDERewriter(llm_model=llm_model) if rw.enable_hyde else None
        multi = (
            MultiQueryExpander(
                llm_model=llm_model,
                num_queries=rw.multi_query_count,
            )
            if rw.enable_multi_query
            else None
        )
        llm_pipeline = QueryRewriterPipeline(
            hyde_rewriter=hyde,
            multi_query_expander=multi,
            enable_hyde=rw.enable_hyde,
            enable_multi_query=rw.enable_multi_query,
        )
        if not llm_pipeline.is_enabled():
            llm_pipeline = None

    if rule is None and llm_pipeline is None:
        return None

    combined = CombinedQueryRewriter(
        rule_rewriter=rule,
        llm_pipeline=llm_pipeline,
        llm_rewrite_once=rw.llm_rewrite_once,
        rewrite_config=rw,
    )
    return combined if combined.is_enabled() else None


def resolve_rewrite_llm(models: Any, config: RagPipelineConfig) -> Optional[Any]:
    rw = config.rewrite
    if not rw.needs_llm_capability():
        return None
    try:
        return models.get_model("router_llm")
    except (KeyError, ValueError, RuntimeError):
        try:
            return models.get_model("main_llm")
        except (KeyError, ValueError, RuntimeError):
            return None
