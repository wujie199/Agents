from typing import List, Optional, Any, TYPE_CHECKING
import logging

from core.ports.rag.rewrite import QueryRewritePort
from document.rag.application.retrieval.rewrite.llm_invoke import invoke_llm_prompt

if TYPE_CHECKING:
    from document.rag.application.retrieval.rewrite.hyde import HyDERewriter


class MultiQueryExpander:
    """Expand one query into multiple LLM-generated variants and fuse results."""

    def __init__(
        self,
        llm_model: Any,
        num_queries: int = 3,
        prompt_template: Optional[str] = None,
    ):
        self._llm = llm_model
        self._num_queries = num_queries
        self._logger = logging.getLogger("rag.rewrite.multiquery")

        self._prompt_template = prompt_template or """你是一个 AI 助手，帮助生成多个相关的检索查询。
给定原始查询，请生成 {num_queries} 个从不同角度表达的相似查询。
每个查询一行，不要编号，不要解释。

原始查询：{query}

扩展查询："""

    async def expand(self, query: str) -> List[str]:
        try:
            prompt = self._prompt_template.format(
                query=query,
                num_queries=self._num_queries,
            )
            response_text = await invoke_llm_prompt(self._llm, prompt)
            if not response_text:
                return [query]

            expanded = [
                line.strip()
                for line in response_text.strip().split("\n")
                if line.strip()
            ]

            all_queries = [query] + expanded[: self._num_queries]
            return all_queries

        except Exception as e:
            self._logger.warning("Multi-query expansion failed, fallback original: %s", e)
            return [query]

    async def expand_batch(self, queries: List[str]) -> List[List[str]]:
        import asyncio

        tasks = [self.expand(q) for q in queries]
        return await asyncio.gather(*tasks)


class QueryRewriterPipeline(QueryRewritePort):
    """Pipeline combining HyDE and multi-query expansion."""

    def __init__(
        self,
        hyde_rewriter: Optional[Any] = None,
        multi_query_expander: Optional[MultiQueryExpander] = None,
        enable_hyde: bool = False,
        enable_multi_query: bool = False,
    ):
        self._hyde = hyde_rewriter
        self._multi_query = multi_query_expander
        self._enable_hyde = enable_hyde
        self._enable_multi_query = enable_multi_query
        self._logger = logging.getLogger("rag.rewrite.pipeline")

    async def rewrite(self, query: str) -> List[str]:
        queries = [query]

        if self._enable_multi_query and self._multi_query:
            queries = await self._multi_query.expand(query)
            self._logger.info(f"Multi-query expanded to {len(queries)} queries")

        if self._enable_hyde and self._hyde:
            rewritten_queries = []
            for q in queries:
                hyde_text = await self._hyde.rewrite(q)
                rewritten_queries.append(hyde_text)
            queries = rewritten_queries
            self._logger.info("HyDE rewriting applied")

        return queries

    async def rewrite_batch(self, queries: List[str]) -> List[List[str]]:
        import asyncio

        tasks = [self.rewrite(q) for q in queries]
        return await asyncio.gather(*tasks)

    def is_enabled(self) -> bool:
        return self._enable_hyde or self._enable_multi_query
