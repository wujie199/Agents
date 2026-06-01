"""RAG utils package exports.

Keep imports minimal to avoid executing heavy I/O at package import time.
"""
from .data_cleaner import (
    clean_text as rag_clean_text,
    postprocess_ocr as rag_postprocess_ocr,
    extract_table_text as rag_extract_table_text,
    normalize_metadata as rag_normalize_metadata,
    dedupe_chunks as rag_dedupe_chunks,
    batch_clean as rag_batch_clean,
)
from .text_chunker import split_text_into_chunks

__all__ = [
    "split_text_into_chunks",
    "rag_clean_text",
    "rag_postprocess_ocr",
    "rag_extract_table_text",
    "rag_normalize_metadata",
    "rag_dedupe_chunks",
    "rag_batch_clean",
]
