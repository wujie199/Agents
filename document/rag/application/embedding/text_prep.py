"""Step 1：进编码器前的文本准备。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Sequence

from document.rag.config.embedding import EmbeddingConfig

_log = logging.getLogger("rag.embedding.text_prep")

EmbedMode = Literal["doc", "query"]

DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


@dataclass(frozen=True)
class PreparedText:
    original_index: int
    text: str


@dataclass
class PrepareResult:
    """非 skipped 项，按 original_index 顺序。"""

    items: List[PreparedText] = field(default_factory=list)
    skipped: List[tuple[int, str]] = field(default_factory=list)

    @property
    def texts(self) -> List[str]:
        return [item.text for item in self.items]

    @property
    def indices(self) -> List[int]:
        return [item.original_index for item in self.items]


def _apply_instruction(text: str, cfg: EmbeddingConfig, mode: EmbedMode) -> str:
    if mode == "query":
        prefix = (cfg.query_instruction or DEFAULT_QUERY_INSTRUCTION).strip()
        if prefix and not text.startswith(prefix):
            return f"{prefix}{text}"
        return text
    doc_prefix = (cfg.doc_instruction or "").strip()
    if doc_prefix and not text.startswith(doc_prefix):
        return f"{doc_prefix}{text}"
    return text


def _truncate_by_chars(text: str, max_chars: int, marker: str) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(1, max_chars - len(marker))
    return text[:keep] + marker


def truncate_text(
    text: str,
    cfg: EmbeddingConfig,
    *,
    tokenize_fn: Optional[Callable[[str], int]] = None,
    decode_truncate_fn: Optional[Callable[[str, int], str]] = None,
) -> str:
    """按 token 数截断；无 tokenizer 时按字符粗截断。"""
    marker = cfg.truncate_marker or "[...]"
    max_tokens = max(8, int(cfg.max_tokens))
    if tokenize_fn and decode_truncate_fn:
        length = tokenize_fn(text)
        if length <= max_tokens - 1:
            return text
        _log.warning(
            "文本超长 (%d tokens > %d)，截断: %r",
            length,
            max_tokens - 1,
            text[:40],
        )
        return decode_truncate_fn(text, max_tokens - 1) + marker
    # 中文粗估：约 1.5 字符/token
    max_chars = max(32, (max_tokens - 1) * 2)
    if len(text) > max_chars:
        _log.warning("文本超长 (chars=%d)，字符级截断", len(text))
        return _truncate_by_chars(text, max_chars, marker)
    return text


def prepare_texts(
    texts: Sequence[str],
    cfg: EmbeddingConfig,
    mode: EmbedMode,
    *,
    tokenize_fn: Optional[Callable[[str], int]] = None,
    decode_truncate_fn: Optional[Callable[[str, int], str]] = None,
) -> PrepareResult:
    """过滤空串、加 instruction、截断。"""
    result = PrepareResult()
    for idx, raw in enumerate(texts):
        text = (raw or "").strip()
        if not text:
            result.skipped.append((idx, "empty"))
            continue
        text = _apply_instruction(text, cfg, mode)
        text = truncate_text(
            text,
            cfg,
            tokenize_fn=tokenize_fn,
            decode_truncate_fn=decode_truncate_fn,
        )
        if not text.strip():
            result.skipped.append((idx, "empty_after_prep"))
            continue
        result.items.append(PreparedText(original_index=idx, text=text))
    return result
