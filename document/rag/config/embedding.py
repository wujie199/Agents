from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding 适配器 + 五步向量化流水线参数。"""

    backend: str = "local_bge"
    model_path: Optional[str] = None
    device: Optional[str] = None
    normalize: bool = True

    # Step 1
    max_tokens: int = 512
    truncate_marker: str = "[...]"
    query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    doc_instruction: str = ""

    # Step 2
    batch_size: int = 32
    batch_size_min: int = 1
    oom_halve_retry: bool = True

    # Step 3
    force_l2_normalize: bool = True
    verify_unit_norm: bool = True
    unit_norm_tolerance: float = 0.02
    reject_zero_vectors: bool = True

    # Step 4（bge-small 不支持 Matryoshka，保留字段供未来扩展）
    matryoshka_dim: Optional[int] = None

    # Step 5
    write_max_retries: int = 3
    dlq_path: str = "data/rag_offline/embedding_dlq.jsonl"

    # P2：增量与迁移
    enable_chunk_incremental: bool = True
    enable_embedding_cache_read: bool = True
    incremental_on_reindex: bool = True
    force_full_delete_on_reindex: bool = False
    versioned_collection: bool = False
