import hashlib
import math
from typing import List


class MockEmbeddingModel:
    """Deterministic fake embeddings for tests and dev without API keys."""

    def __init__(self, dimension: int = 64):
        self._dimension = dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._vectorize(text) for text in texts]

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)

    def _vectorize(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for i in range(self._dimension):
            b = digest[i % len(digest)]
            values.append((b / 127.5) - 1.0)
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]
