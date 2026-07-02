from typing import List, Optional, Any
import logging

from document.rag.application.retrieval.rewrite.llm_invoke import invoke_llm_prompt


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
                hypothesis = await invoke_llm_prompt(self._llm, prompt)
                if hypothesis:
                    hypotheses.append(hypothesis.strip())
                else:
                    hypotheses.append(query)

            except Exception as e:
                self._logger.warning("Hypothesis generation failed, fallback original: %s", e)
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
