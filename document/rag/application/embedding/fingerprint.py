"""Chunk 向量化内容指纹（用于增量跳过重编码）。"""

from __future__ import annotations

import hashlib


def chunk_embed_fingerprint(prepared_text: str, *, model_version: str) -> str:
    """Step1 后的待编码文本 + model_version → 稳定指纹。"""
    payload = f"{model_version}\n{prepared_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
