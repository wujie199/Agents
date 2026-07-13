"""分句与文本工具。"""

import re
from typing import List, Tuple

from document.rag.application.chunking.models import SentenceSpan, StructuralUnit
from document.rag.shared.data_cleaner import tokenize_text


_SECTION_NUMBER_DOT = re.compile(
    r"(?<=\d)\.(?=\d)|(?<=[第条章节])\d+\.(?=\d)"
)
_SENT_END_ZH = re.compile(r"(?<=[。！？；])")
_SENT_END_EN = re.compile(r"(?<=[.!?;])\s+")

_CN_PRONOUNS = re.compile(
    r"(它|该|此|其|上述|该系统|该方法|这|那|前者|后者|它们|这些|那些)"
)
_DEFINITION = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9_\-]+(?:是指|定义为|是一种|指的是)"
)
_EXAMPLE = re.compile(r"(例如|比如|如：|如下所示|举例)")


def protect_section_numbers(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    replacements: List[Tuple[str, str]] = []

    def _repl(match: re.Match) -> str:
        token = f"§SEC{len(replacements)}§"
        replacements.append((token, match.group(0)))
        return token

    protected = _SECTION_NUMBER_DOT.sub(_repl, text)
    return protected, replacements


def restore_section_numbers(text: str, replacements: List[Tuple[str, str]]) -> str:
    out = text
    for token, original in replacements:
        out = out.replace(token, original)
    return out


def split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    protected, repl = protect_section_numbers(text.strip())
    parts: List[str] = []
    if re.search(r"[\u4e00-\u9fff]", protected):
        raw = [p for p in _SENT_END_ZH.split(protected) if p.strip()]
        parts = [restore_section_numbers(p.strip(), repl) for p in raw if p.strip()]
    else:
        raw = [p for p in _SENT_END_EN.split(protected) if p.strip()]
        parts = [restore_section_numbers(p.strip(), repl) for p in raw if p.strip()]
    if not parts:
        parts = [restore_section_numbers(protected, repl)]
    return parts


def flatten_unit_sentences(units: List[StructuralUnit]) -> List[SentenceSpan]:
    spans: List[SentenceSpan] = []
    global_idx = 0
    offset = 0
    for u_idx, unit in enumerate(units):
        if unit.unit_type in ("table", "list", "code", "qa"):
            spans.append(
                SentenceSpan(
                    text=unit.content,
                    unit_index=u_idx,
                    global_index=global_idx,
                    char_start=offset,
                    char_end=offset + len(unit.content),
                )
            )
            global_idx += 1
            offset += len(unit.content) + 1
            continue
        sentences = split_sentences(unit.content)
        if not sentences:
            continue
        merged: List[str] = []
        for sent in sentences:
            if merged and len(sent) < 8:
                merged[-1] = merged[-1] + sent
            else:
                merged.append(sent)
        for sent in merged:
            spans.append(
                SentenceSpan(
                    text=sent,
                    unit_index=u_idx,
                    global_index=global_idx,
                    char_start=offset,
                    char_end=offset + len(sent),
                )
            )
            global_idx += 1
            offset += len(sent)
    return spans


def keyword_set(text: str) -> set:
    return set(tokenize_text(text))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
