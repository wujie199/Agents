"""Recovered docstring.

拆分后的精简版：文本清理 + batch 处理 + re-export 兼容层。
表格提取 → table_parsing.py；去重 → dedupe.py；元数据规范化 → metadata_norm.py
"""
from datetime import datetime
from typing import List, Dict, Any, Generator, Iterable, Optional, Tuple, Callable, Union
from collections import Counter
import logging
import re
import unicodedata
from hashlib import md5

# Language detection: langdetect if installed, else heuristic
try:
    from langdetect import detect as _ld_detect
except ImportError:
    _ld_detect = None

LOGGER = logging.getLogger(__name__)
_DATA_CLEANER_METRICS: Counter = Counter()


def _increment_metric(name: str, amount: int = 1) -> None:
    _DATA_CLEANER_METRICS[name] += amount


def get_data_cleaner_metrics() -> Dict[str, int]:
    return dict(_DATA_CLEANER_METRICS)


def reset_data_cleaner_metrics() -> None:
    _DATA_CLEANER_METRICS.clear()


# ── 文本清理 ──────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Basic text cleanup."""
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r"[\x00-\x1f\x7f]+", " ", s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _safe_call(fn: Callable[..., Any], *args: Any, fallback: Any = None, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except (RuntimeError, ValueError, OSError):
        LOGGER.exception("Data cleaner failed inside %s", fn.__name__)
        _increment_metric(f"{fn.__name__}_errors")
        return fallback


def _merge_ocr_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    merged = []
    for line in lines:
        if not line:
            continue
        if not merged:
            merged.append(line)
            continue
        prev = merged[-1]
        sentence_end = prev.endswith((".", "。", "！", "？", "!", "?"))
        if sentence_end or re.match(r"^[A-Z]", line) or re.match(r"^[\u4e00-\u9fff]", line):
            merged.append(line)
        else:
            merged[-1] = prev + " " + line
    return "\n".join(merged)


def _clean_preserved_lines(text: str) -> str:
    lines = [clean_text(line) for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _strip_ocr_noise(text: str) -> str:
    s = text.replace("\ufeff", "").replace("\ufffd", "")
    s = re.sub(r"-\s*\n\s*", "", s)
    s = re.sub(r"[|\u2500-\u257f]+", " ", s)
    s = re.sub(r"\.{2,}", ".", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def ocr_confidence_score(text: str) -> float:
    """Estimate OCR confidence from text noise heuristics."""
    if not text:
        return 0.0
    s = str(text)
    allowed = re.compile(
        r"[A-Za-z0-9\u4e00-\u9fff\s\.,!;:'\"\-\(\)\[\]\{\}\\/]+"
    )
    invalid = sum(1 for ch in s if not allowed.match(ch))
    ratio = invalid / len(s)
    bad_repeat = len(re.findall(r"(.)\1{4,}", s))
    score = max(0.0, 1.0 - min(0.8, ratio + bad_repeat * 0.05))
    return round(score, 3)


def _is_table_like_text(text: str) -> bool:
    if not text or text.count("\n") < 2:
        return False
    if "|" in text or "\t" in text:
        return True
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    separators = sum(1 for line in lines if re.search(r"\s{2,}", line))
    return separators >= max(1, len(lines) // 2)


def _normalize_table_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def postprocess_ocr(
    text: str,
    bbox: Dict[str, Any] = None,
    preserve_layout: bool = False,
    preserve_tables: bool = True,
) -> str:
    """Post-process OCR output: layout repair and table/noise handling."""
    if text is None:
        return ""
    s = str(text)
    table_like = preserve_tables and _is_table_like_text(s)
    s = _strip_ocr_noise(s)
    if table_like:
        return _clean_preserved_lines(_normalize_table_text(s))
    if preserve_layout:
        return _clean_preserved_lines(_merge_ocr_lines(s))
    return clean_text(s.replace("\n", " "))


def detect_language(text: str) -> str:
    """Detect language of `text`.

    Returns an ISO 639-1 code like 'en', 'zh', or 'und' for undetermined.
    Uses `langdetect` when available; otherwise a simple CJK/ASCII heuristic.
    """
    if not text:
        return "und"
    s = str(text)
    if _ld_detect:
        try:
            return _ld_detect(s)
        except (ValueError, RuntimeError):
            pass
    if re.search(r"[\u4e00-\u9fff]", s):
        return "zh"
    if re.search(r"[A-Za-z]", s):
        return "en"
    return "und"


def tokenize_text(text: str) -> List[str]:
    """Lightweight tokenizer: whitespace + punctuation split.

    For production you may replace this with `jieba`, `spacy`, or a BPE tokenizer.
    """
    if not text:
        return []
    s = re.sub(r"\s+", " ", str(text)).strip()
    parts = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", s)
    return parts


# ── Batch 处理 ────────────────────────────────────────────


def _chunk_hash(chunk_text: str) -> str:
    return md5(chunk_text.encode("utf-8")).hexdigest()


def _safe_clean_record(rec: Any) -> Optional[Dict[str, Any]]:
    try:
        if not isinstance(rec, dict):
            raise ValueError("record must be a dict")
        item = _clean_record(rec)
        _increment_metric("records_cleaned")
        return item
    except (RuntimeError, ValueError, OSError, TypeError):
        LOGGER.exception("Failed to clean record: %r", rec)
        _increment_metric("record_errors")
        return None


def _clean_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    from document.rag.shared.metadata_norm import normalize_metadata

    raw = rec.get("raw_text") or rec.get("text") or ""
    text = clean_text(raw)
    metadata = normalize_metadata(rec.get("meta") or {})
    lang = detect_language(text)
    tokens = tokenize_text(text)
    rid = _chunk_hash((rec.get("source_id", "") + "||" + text)[:1024])
    return {
        "id": rid,
        "chunk_text": text,
        "metadata": metadata,
        "source": rec.get("source"),
        "lang": lang,
        "tokens": tokens,
        "token_count": len(tokens),
    }


def batch_clean(records: Iterable[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
    """Batch clean records (compat API)."""
    for rec in records:
        item = _safe_clean_record(rec)
        if item is not None:
            yield item


def batch_clean_parallel(
    records: Iterable[Dict[str, Any]],
    workers: int = 4,
    use_process: bool = False,
    chunk_size: int = 64,
) -> Generator[Dict[str, Any], None, None]:
    """Parallel batch clean."""
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    from itertools import islice

    Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor
    it = iter(records)

    with Executor(max_workers=workers) as ex:
        futures = []

        while True:
            chunk = list(islice(it, chunk_size))
            if not chunk:
                break
            for rec in chunk:
                futures.append(ex.submit(_safe_clean_record, rec))
            if len(futures) >= workers * 4:
                for fut in futures:
                    try:
                        item = fut.result()
                    except (RuntimeError, ValueError, OSError, TypeError):
                        LOGGER.exception("Parallel worker failed")
                        _increment_metric("parallel_errors")
                        continue
                    if item is not None:
                        yield item
                futures = []

        for fut in futures:
            try:
                item = fut.result()
            except (RuntimeError, ValueError, OSError, TypeError):
                LOGGER.exception("Parallel worker failed")
                _increment_metric("parallel_errors")
                continue
            if item is not None:
                yield item


# ── Re-export 兼容层 ──────────────────────────────────────

from document.rag.shared.table_parsing import extract_table_text  # noqa: E402
from document.rag.shared.dedupe import dedupe_chunks, semantic_dedupe  # noqa: E402
from document.rag.shared.metadata_norm import normalize_metadata  # noqa: E402

__all__ = [
    "clean_text",
    "postprocess_ocr",
    "ocr_confidence_score",
    "extract_table_text",
    "normalize_metadata",
    "dedupe_chunks",
    "semantic_dedupe",
    "batch_clean",
    "batch_clean_parallel",
    "detect_language",
    "tokenize_text",
    "get_data_cleaner_metrics",
    "reset_data_cleaner_metrics",
]
