from typing import List


class TruncatingSummarizerAdapter:
    """默认摘要：拼接片段并按字符预算截断（不依赖 LLM）。"""

    def __init__(self, max_chars: int = 2000):
        self._max_chars = max_chars

    async def summarize(self, fragments: List[str], query: str) -> str:
        if not fragments:
            return "No relevant messages found"
        combined = "\n\n".join(fragments)
        if len(combined) <= self._max_chars:
            return combined
        truncated = combined[: self._max_chars]
        last_break = max(truncated.rfind("\n\n"), truncated.rfind("\n"))
        if last_break > self._max_chars * 0.7:
            truncated = truncated[:last_break]
        return truncated + "\n[... truncated ...]"
