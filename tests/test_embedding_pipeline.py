"""五步向量化流水线单元测试（无需真实 bge 权重）。"""

from __future__ import annotations

import math
from typing import List

import pytest

from document.rag.application.embedding.batch_encode import encode_batches
from document.rag.application.embedding.encoder import EmbeddingEncoder
from document.rag.application.embedding.normalizer import normalize_vector
from document.rag.application.embedding.text_prep import prepare_texts
from document.rag.application.indexing.embedder import Embedder
from document.rag.config.embedding import EmbeddingConfig
from core.ports.chunker import Chunk


class _MockEmbedding:
    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)


class _ZeroThenOkEmbedding:
    def __init__(self):
        self.calls = 0

    def embed(self, texts: List[str]) -> List[List[float]]:
        self.calls += 1
        if self.calls == 1:
            return [[0.0, 0.0, 0.0] for _ in texts]
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)


@pytest.fixture
def emb_cfg() -> EmbeddingConfig:
    return EmbeddingConfig(batch_size=2, max_tokens=512)


def test_prepare_texts_filters_empty(emb_cfg):
    result = prepare_texts(["", "  ", "有效文本"], emb_cfg, "doc")
    assert len(result.items) == 1
    assert result.items[0].text == "有效文本"
    assert len(result.skipped) == 2


def test_prepare_texts_query_instruction(emb_cfg):
    result = prepare_texts(["什么是RAG"], emb_cfg, "query")
    assert result.texts[0].startswith("为这个句子生成表示以用于检索相关文章：")
    assert "什么是RAG" in result.texts[0]


def test_prepare_texts_doc_no_instruction(emb_cfg):
    result = prepare_texts(["合同条款"], emb_cfg, "doc")
    assert result.texts[0] == "合同条款"


def test_normalize_rejects_zero_vector(emb_cfg):
    assert normalize_vector([0.0, 0.0], emb_cfg) is None


def test_normalize_unit_vector(emb_cfg):
    vec = normalize_vector([3.0, 4.0], emb_cfg)
    assert vec is not None
    norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    assert abs(norm - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_encode_batches_respects_batch_size(emb_cfg):
    model = _MockEmbedding()
    cfg = EmbeddingConfig(batch_size=2)
    vectors = await encode_batches(model, ["a", "bb", "ccc", "dddd"], cfg)
    assert len(vectors) == 4


@pytest.mark.asyncio
async def test_encoder_skips_zero_vectors(emb_cfg):
    model = _ZeroThenOkEmbedding()
    encoder = EmbeddingEncoder(model, emb_cfg)
    vectors, prepared = await encoder.encode_texts(["hello"], "doc")
    assert vectors == []
    assert prepared.skipped


@pytest.mark.asyncio
async def test_embedder_doc_pipeline(emb_cfg):
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", content="第一段", chunk_index=0),
        Chunk(chunk_id="c2", doc_id="d1", content="   ", chunk_index=1),
    ]
    embedder = Embedder(_MockEmbedding(), embedding_cfg=emb_cfg, enable_cache=False)
    result = await embedder.embed_chunks(chunks, "tenant-a")
    assert len(result.to_write) == 1
    assert result.to_write[0]["chunk_id"] == "c1"
    assert len(result.to_write[0]["embedding"]) == 3


@pytest.mark.asyncio
async def test_encoder_query(emb_cfg):
    encoder = EmbeddingEncoder(_MockEmbedding(), emb_cfg)
    vec = await encoder.encode_query("测试问题")
    assert len(vec) == 3
