"""本地 BAAI/bge-small-zh-v1.5 向量（sentence-transformers）。"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from document.model_mount import (
    DEFAULT_EMBEDDING_MODEL,
    require_mounted_volume,
    unmounted_reminder,
)

_log = logging.getLogger("document.rag.components.embedding.local_bge")


class LocalBgeEmbedding:
    """使用本地 bge-small-zh-v1.5 生成文本向量；接口与 Embedder / MockEmbeddingModel 一致。"""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        if model_dir:
            resolved = Path(model_dir).expanduser()
            if not resolved.is_absolute():
                resolved = (Path.cwd() / resolved).resolve()
            else:
                resolved = resolved.resolve()
            self._model_dir = str(resolved)
        else:
            self._model_dir = str(DEFAULT_EMBEDDING_MODEL)

        require_mounted_volume(
            self._model_dir,
            purpose="RAG 本地 embedding（bge-small-zh-v1.5）",
            env_hint="可在 config/rag_pipeline.yml 的 embedding.model_path 或 RAG_EMBEDDING_MODEL 指定路径",
        )

        model_path = Path(self._model_dir)
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"本地 embedding 目录不完整: {self._model_dir}\n"
                + unmounted_reminder(
                    path=self._model_dir,
                    purpose="RAG embedding 权重缺失",
                    env_hint="embedding.model_path",
                )
            )

        self._normalize = normalize_embeddings
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        self._model = SentenceTransformer(self._model_dir, **kwargs)
        _log.info("已加载本地 embedding: %s", self._model_dir)

    @property
    def model_dir(self) -> str:
        return self._model_dir

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed, texts)
