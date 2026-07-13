"""五步向量化编排（Step1–3；Step4 Matryoshka 对 bge-small 为 no-op）。"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional, Sequence

from core.ports.rag.embedding import EmbeddingPort
from document.rag.application.embedding.batch_encode import encode_batches
from document.rag.application.embedding.normalizer import normalize_vectors
from document.rag.application.embedding.text_prep import EmbedMode, PrepareResult, prepare_texts
from document.rag.config.embedding import EmbeddingConfig

_log = logging.getLogger("rag.embedding.encoder")


class EmbeddingEncoder:
    """统一 doc / query 编码入口。"""

    def __init__(
        self,
        model: EmbeddingPort,
        cfg: EmbeddingConfig,
    ):
        self._model = model
        self._cfg = cfg
        self._tokenize_fn = None
        self._decode_truncate_fn = None
        if hasattr(model, "token_length"):
            self._tokenize_fn = model.token_length  # type: ignore[attr-defined]
        if hasattr(model, "truncate_to_tokens"):
            self._decode_truncate_fn = model.truncate_to_tokens  # type: ignore[attr-defined]

    @property
    def config(self) -> EmbeddingConfig:
        return self._cfg

    async def encode_texts(
        self,
        texts: Sequence[str],
        mode: EmbedMode,
    ) -> tuple[List[List[float]], PrepareResult]:
        """返回 (vectors, prepare_result)；vectors 与 prepare_result.items 一一对应。"""
        prepared = prepare_texts(
            texts,
            self._cfg,
            mode,
            tokenize_fn=self._tokenize_fn,
            decode_truncate_fn=self._decode_truncate_fn,
        )
        if not prepared.texts:
            return [], prepared
        raw_vectors = await encode_batches(self._model, prepared.texts, self._cfg)
        if len(raw_vectors) != len(prepared.items):
            raise RuntimeError(
                f"编码数量不匹配: prepared={len(prepared.items)} vectors={len(raw_vectors)}"
            )
        final_vectors: List[List[float]] = []
        final_items = []
        from document.rag.application.embedding.normalizer import normalize_vector

        for item, vec in zip(prepared.items, raw_vectors):
            normalized = normalize_vector(vec, self._cfg)
            if normalized is None:
                prepared.skipped.append((item.original_index, "zero_vector"))
                _log.warning(
                    "index=%s 向量为零，跳过", item.original_index
                )
                continue
            final_items.append(item)
            final_vectors.append(normalized)
        prepared.items = final_items
        return final_vectors, prepared

    async def encode_query(self, text: str) -> List[float]:
        vectors, prepared = await self.encode_texts([text], "query")
        if not vectors:
            raise ValueError("query 为空或编码失败")
        if prepared.skipped:
            _log.warning("query 编码跳过: %s", prepared.skipped)
        return vectors[0]

    async def encode_queries(self, texts: Sequence[str]) -> List[List[float]]:
        vectors, _ = await self.encode_texts(texts, "query")
        return vectors
