"""本地 BAAI/bge-small-zh-v1.5 向量（sentence-transformers）。"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

_log = logging.getLogger("document.rag.adapters.embedding.local_bge")

_DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "weights" / "bge-small-zh-v1.5"
)


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
            self._model_dir = str(_DEFAULT_MODEL_DIR)

        model_path = Path(self._model_dir)
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"本地 embedding 目录不完整: {self._model_dir}\n"
                "请确认存在 config.json 与 model.safetensors（或 pytorch_model.bin）"
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
