import hashlib
import logging

from agent_platform.memory.adapters.summarizer_adapter import TruncatingSummarizerAdapter


def _extract_llm_text(response) -> str:
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        return (getattr(msg, "content", None) or "").strip()
    if isinstance(response, str):
        return response.strip()
    return ""


class LlmMemorySummarizerAdapter:
    """L2 session_search 二次摘要：独立 LLM role，失败降级截断。"""

    def __init__(
        self,
        models=None,
        role: str = "memory_summarizer_llm",
        max_chars: int = 2000,
        fallback: TruncatingSummarizerAdapter | None = None,
    ):
        self._models = models
        self._role = role
        self._max_chars = max_chars
        self._fallback = fallback or TruncatingSummarizerAdapter(max_chars=max_chars)
        self._logger = logging.getLogger(__name__)

    async def summarize(self, fragments: list[str], query: str) -> str:
        if not fragments:
            return "No relevant messages found"
        if self._models is None:
            return await self._fallback.summarize(fragments, query)

        try:
            model = self._models.get_model(self._role)
            body = "\n\n".join(fragments)
            response = await model.ainvoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "Summarize the session fragments relevant to the query. "
                            "Be concise; preserve names, dates, and decisions."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nFragments:\n{body}",
                    },
                ]
            )
            text = _extract_llm_text(response)
            if text:
                if len(text) > self._max_chars:
                    return await self._fallback.summarize([text], query)
                return text
        except Exception as e:
            self._logger.warning("L2 LLM summarize failed, fallback truncate: %s", e)

        return await self._fallback.summarize(fragments, query)


class CachedMemorySummarizerAdapter:
    """包装任意 Summarizer，对相同 query+fragments 做进程内缓存。"""

    def __init__(self, inner, max_entries: int = 256):
        self._inner = inner
        self._cache: dict[str, str] = {}
        self._max_entries = max_entries

    @staticmethod
    def _key(fragments: list[str], query: str) -> str:
        raw = query + "\n" + "\n".join(fragments)
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    async def summarize(self, fragments: list[str], query: str) -> str:
        key = self._key(fragments, query)
        if key in self._cache:
            return self._cache[key]
        result = await self._inner.summarize(fragments, query)
        if len(self._cache) >= self._max_entries:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = result
        return result
