from typing import Protocol


class HotMemoryCompressor(Protocol):
    """L1 超预算压缩契约（LLM 或截断）。"""

    async def compress_memory(self, content: str, max_chars: int) -> str:
        ...

    async def compress_user(self, content: str, max_chars: int) -> str:
        ...
