"""BM25 索引工厂。"""

from pathlib import Path

from document.rag.adapters.retrieval.bm25_local import LocalBm25Index
from document.rag.config import RagPipelineConfig


def build_bm25_index(
    data_dir: Path,
    cfg: RagPipelineConfig,
) -> LocalBm25Index:
    return LocalBm25Index.for_collection(data_dir, cfg.collection_name)
