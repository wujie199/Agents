"""离线查询栈：Chroma + BM25 + 混合检索 + Rerank。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from document.rag.components.embedding.registry import build_embedding
from document.rag.components.rerank.registry import build_rerank
from document.rag.components.storage.registry import build_bm25_index
from document.rag.config import RagPipelineConfig, load_rag_pipeline_config

_log = logging.getLogger("document.rag.bootstrap.query")


@dataclass
class QueryStack:
    config: RagPipelineConfig
    vector_port: Any
    embedding_model: Any
    bm25_index: Any
    rerank_model: Any
    chroma_dir: str


def create_query_stack(
    data_dir: Path,
    *,
    config_dir: str = "config",
    cfg: Optional[RagPipelineConfig] = None,
    tenant_id: str = "default",
    auto_rebuild_bm25: bool = True,
) -> QueryStack:
    """
    组装离线索引查询栈。

    data_dir 需与建库一致（如 data/rag_offline），Chroma 在 {data_dir}/chroma_dev。
    """
    from agent_platform.storage.adapters.chroma.vector_adapter import ChromaVectorAdapter

    cfg = cfg or load_rag_pipeline_config(config_dir=config_dir)
    chroma_dir = str(data_dir / "chroma_dev")
    vector_port = ChromaVectorAdapter(persist_directory=chroma_dir)
    embedding = build_embedding(cfg)
    bm25_index = build_bm25_index(data_dir, cfg)
    rerank_model = build_rerank(cfg)

    if auto_rebuild_bm25 and bm25_index.document_count == 0:
        try:
            count = vector_port.count(cfg.collection_name)
        except (RuntimeError, ConnectionError, OSError, ValueError):
            count = 0
        if count > 0:
            _log.warning(
                "BM25 索引为空但 Chroma 有 %d 条向量，正在从 Chroma 重建 BM25",
                count,
            )
            bm25_index.rebuild_from_chroma(
                chroma_dir,
                cfg.collection_name,
                tenant_id=tenant_id,
            )

    return QueryStack(
        config=cfg,
        vector_port=vector_port,
        embedding_model=embedding,
        bm25_index=bm25_index,
        rerank_model=rerank_model,
        chroma_dir=chroma_dir,
    )


def rebuild_bm25_from_chroma(
    data_dir: Path,
    *,
    config_dir: str = "config",
    tenant_id: Optional[str] = None,
    cfg: Optional[RagPipelineConfig] = None,
) -> Tuple[int, str]:
    cfg = cfg or load_rag_pipeline_config(config_dir=config_dir)
    chroma_dir = str(data_dir / "chroma_dev")
    bm25_index = build_bm25_index(data_dir, cfg)
    n = bm25_index.rebuild_from_chroma(
        chroma_dir,
        cfg.collection_name,
        tenant_id=tenant_id,
    )
    return n, str(bm25_index._path)
