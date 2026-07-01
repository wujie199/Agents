"""Storage 独立 registry。"""

from pathlib import Path
from typing import Any

from document.rag.config.pipeline import RagPipelineConfig


def build_bm25_index(data_dir: Path, cfg: RagPipelineConfig) -> Any:
    from document.rag.components.storage.bm25_local import LocalBm25Index
    return LocalBm25Index.for_collection(data_dir, cfg.collection_name)
