"""向量嵌入端口。"""

from typing import List, Protocol


class EmbeddingPort(Protocol):
    """RAG 内部：文本 → 向量嵌入。"""

    def embed(self, texts: List[str]) -> List[List[float]]:
        ...

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        ...
