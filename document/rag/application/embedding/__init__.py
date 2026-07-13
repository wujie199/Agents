"""五步向量化流水线（文本准备 → 批量编码 → 归一化 → 写入）。"""

from document.rag.application.embedding.encoder import EmbeddingEncoder
from document.rag.application.embedding.text_prep import PrepareResult, prepare_texts

__all__ = ["EmbeddingEncoder", "PrepareResult", "prepare_texts"]
