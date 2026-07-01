from typing import List, Optional, Dict, Any, TYPE_CHECKING
import logging

from core.ports.rag.rewrite import QueryRewritePort

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
        
        self._prompt_template = prompt_template or """ä½ æ¯ä¸ä¸ªAIå©æï¼å¸®å©çæå¤ä¸ªç¸å³çæç´¢æ¥è¯¢ã?ç»å®åå§æ¥è¯¢ï¼è¯·çæ {num_queries} ä¸ªä»ä¸åè§åº¦è¡¨è¾¾çç¸ä¼¼æ¥è¯¢ã?æ¯ä¸ªæ¥è¯¢ä¸è¡ï¼ä¸è¦ç¼å·ï¼ä¸è¦è§£éã?
åå§æ¥è¯¢ï¼{query}

æ©å±æ¥è¯¢ï¼"""
    
    async def expand(self, query: str) -> List[str]:
        try:
            prompt = self._prompt_template.format(
                query=query,
                num_queries=self._num_queries
            )
            
            if hasattr(self._llm, 'ainvoke'):
                response = await self._llm.ainvoke(prompt)
            elif hasattr(self._llm, 'invoke'):
                response = self._llm.invoke(prompt)
            else:
                self._logger.warning("LLM has no invoke method")
                return [query]
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            expanded = [
                line.strip()
                for line in response_text.strip().split('\n')
                if line.strip()
            ]
            
            all_queries = [query] + expanded[:self._num_queries]
            
            return all_queries
            
        except (RuntimeError, ValueError, TimeoutError, ConnectionError) as e:
            self._logger.error(f"Multi-query expansion failed: {e}")
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
