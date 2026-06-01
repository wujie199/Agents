"""本地 bge-small-zh-v1.5 embedding（需权重目录完整）。"""

import pytest

from document.rag.adapters.embedding.local_bge import LocalBgeEmbedding


@pytest.fixture
def embedder():
    pytest.importorskip("sentence_transformers")
    return LocalBgeEmbedding(device="cpu")


def test_embed_dimensions(embedder):
    vectors = embedder.embed(["你好", "世界"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert len(vectors[0]) > 64


@pytest.mark.asyncio
async def test_aembed(embedder):
    vectors = await embedder.aembed(["async test"])
    assert len(vectors) == 1
    assert len(vectors[0]) > 0
