from typing import List, Optional, Dict, Any
import logging


class HyDERewriter:
    """HyDE: generate hypothetical document, embed it for retrieval."""
    
    def __init__(
        self,
        llm_model: Any,
        prompt_template: Optional[str] = None,
        num_hypotheses: int = 1,
    ):
        self._llm = llm_model
        self._num_hypotheses = num_hypotheses
        self._logger = logging.getLogger("rag.rewrite.hyde")
        
        self._prompt_template = prompt_template or """请根据以下问题，写一段可能包含答案的文档片段。不要解释，直接输出文档内容。
问题：{query}

文档片段："""
    
    async def rewrite(self, query: str) -> str:
        hypotheses = await self._generate_hypotheses(query)
        
        if len(hypotheses) == 1:
            return hypotheses[0]
        
        return "\n\n".join(hypotheses)
    
    async def rewrite_batch(self, queries: List[str]) -> List[str]:
        import asyncio
        tasks = [self.rewrite(q) for q in queries]
        return await asyncio.gather(*tasks)
    
    async def _generate_hypotheses(self, query: str) -> List[str]:
        hypotheses = []
        
        for _ in range(self._num_hypotheses):
            try:
                prompt = self._prompt_template.format(query=query)
                
                if hasattr(self._llm, 'ainvoke'):
                    response = await self._llm.ainvoke(prompt)
                elif hasattr(self._llm, 'invoke'):
                    response = self._llm.invoke(prompt)
                else:
                    self._logger.warning("LLM has no invoke method")
                    return [query]
                
                hypothesis = response.content if hasattr(response, 'content') else str(response)
                hypotheses.append(hypothesis.strip())
                
            except (RuntimeError, ValueError, TimeoutError, ConnectionError) as e:
                self._logger.error(f"Hypothesis generation failed: {e}")
                hypotheses.append(query)
        
        return hypotheses if hypotheses else [query]


class HyDEAdapter:
    """HyDE adapter for retrieval pipeline integration."""
    
    def __init__(self, rewriter: HyDERewriter):
        self._rewriter = rewriter
    
    async def get_search_text(self, query: str) -> str:
        return await self._rewriter.rewrite(query)
    
    async def get_search_texts(self, queries: List[str]) -> List[str]:
        return await self._rewriter.rewrite_batch(queries)
