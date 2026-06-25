"""L2 检索片段摘要契约。"""

from typing import Protocol, List


class MemorySummarizer(Protocol):
    """L2 检索片段摘要契约（默认截断，可替换为 LLM 实现）。"""

    async def summarize(self, fragments: List[str], query: str) -> str:
        ...
