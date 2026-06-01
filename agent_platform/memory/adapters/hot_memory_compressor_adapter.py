import logging

from agent_platform.memory.adapters.hot_memory_file_adapter import HotMemoryFileAdapter
from agent_platform.memory.adapters.summarizer_adapter import TruncatingSummarizerAdapter


class TruncatingHotMemoryCompressorAdapter:
    """L1 压缩：按段落截断（无 LLM）。"""

    async def compress_memory(self, content: str, max_chars: int) -> str:
        return HotMemoryFileAdapter.truncate(content, max_chars)

    async def compress_user(self, content: str, max_chars: int) -> str:
        return HotMemoryFileAdapter.truncate(content, max_chars)


def _extract_llm_text(response) -> str:
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        return (getattr(msg, "content", None) or "").strip()
    if isinstance(response, str):
        return response.strip()
    return ""


class LlmHotMemoryCompressorAdapter:
    """L1 压缩：LLM 摘要，失败降级为截断。"""

    def __init__(
        self,
        models=None,
        role: str = "memory_summarizer_llm",
        fallback: TruncatingHotMemoryCompressorAdapter | None = None,
    ):
        self._models = models
        self._role = role
        self._fallback = fallback or TruncatingHotMemoryCompressorAdapter()
        self._logger = logging.getLogger(__name__)

    async def _compress(self, content: str, max_chars: int, label: str) -> str:
        if len(content) <= max_chars:
            return content
        if self._models is None:
            return await self._fallback.compress_memory(content, max_chars)

        try:
            model = self._models.get_model(self._role)
            response = await model.ainvoke(
                [
                    {
                        "role": "system",
                        "content": (
                            f"Compress the following {label} memory to under "
                            f"{max_chars} characters. Keep key facts as bullet lines."
                        ),
                    },
                    {"role": "user", "content": content},
                ]
            )
            text = _extract_llm_text(response)
            if text and len(text) <= max_chars * 1.1:
                return HotMemoryFileAdapter.truncate(text, max_chars)
        except Exception as e:
            self._logger.warning("L1 LLM compress failed, fallback truncate: %s", e)

        return await self._fallback.compress_memory(content, max_chars)

    async def compress_memory(self, content: str, max_chars: int) -> str:
        return await self._compress(content, max_chars, "system")

    async def compress_user(self, content: str, max_chars: int) -> str:
        return await self._compress(content, max_chars, "user")
