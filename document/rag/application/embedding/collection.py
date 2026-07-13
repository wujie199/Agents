"""按 model_version 解析向量库 collection（模型迁移 A/B）。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from document.rag.config.pipeline import RagPipelineConfig


def slugify_model_version(model_version: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (model_version or "default").strip())
    return slug.strip("_").lower()[:48] or "default"


def resolve_versioned_collection(base: str, model_version: str) -> str:
    """versioned collection 名：{base}_{model_slug}。"""
    return f"{base}_{slugify_model_version(model_version)}"


def resolve_index_collection(
    base: str,
    model_version: str,
    *,
    versioned: bool,
) -> str:
    if not versioned:
        return base
    return resolve_versioned_collection(base, model_version)


def effective_collection_name(cfg: "RagPipelineConfig") -> str:
    """建库与检索共用的实际 Chroma/BM25 collection 名。"""
    return resolve_index_collection(
        cfg.collection_name,
        cfg.model_version,
        versioned=cfg.embedding.versioned_collection,
    )
