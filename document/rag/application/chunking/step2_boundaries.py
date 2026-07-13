"""Step2：语义边界检测（TopicTiling + Embedding TextTiling + 三档仲裁 + 禁忌）。"""

import re
from typing import Callable, List, Optional, Set

from document.rag.config.chunk_pipeline import ChunkPipelineConfig
from document.rag.application.chunking.models import (
    BoundaryResult,
    CutPoint,
    ForbiddenRange,
    SentenceSpan,
    StructuralUnit,
)
from document.rag.application.chunking.text_utils import (
    cosine_similarity,
    jaccard,
    keyword_set,
    _CN_PRONOUNS,
    _DEFINITION,
    _EXAMPLE,
)


def _topic_tiling_cuts(
    sentences: List[SentenceSpan],
    window: int,
    threshold: float,
) -> Set[int]:
    if len(sentences) < 3:
        return set()
    keyword_sets = [keyword_set(s.text) for s in sentences]
    cuts: Set[int] = set()
    for i in range(1, len(sentences)):
        left_start = max(0, i - window)
        right_end = min(len(sentences), i + window)
        left = set()
        right = set()
        for j in range(left_start, i):
            left |= keyword_sets[j]
        for j in range(i, right_end):
            right |= keyword_sets[j]
        overlap = jaccard(left, right)
        if overlap < threshold:
            cuts.add(i)
    return cuts


def _embedding_tiling_cuts(
    sentences: List[SentenceSpan],
    embed_fn: Callable[[List[str]], List[List[float]]],
    drop_threshold: float,
) -> Set[int]:
    if len(sentences) < 3:
        return set()
    texts = [s.text for s in sentences]
    vectors = embed_fn(texts)
    if len(vectors) != len(texts):
        return set()
    sims = [
        cosine_similarity(vectors[i], vectors[i + 1])
        for i in range(len(vectors) - 1)
    ]
    cuts: Set[int] = set()
    for i in range(1, len(sims) - 1):
        left = sims[i - 1]
        cur = sims[i]
        right = sims[i + 1]
        if cur < left and cur < right and (left - cur) >= drop_threshold:
            cuts.add(i + 1)
    return cuts


def _detect_forbidden_ranges(
    sentences: List[SentenceSpan],
    units: List[StructuralUnit],
) -> List[ForbiddenRange]:
    forbidden: List[ForbiddenRange] = []
    n = len(sentences)
    if n == 0:
        return forbidden

    # 结构单元内部禁止切分
    for i in range(n - 1):
        if sentences[i].unit_index == sentences[i + 1].unit_index:
            u = units[sentences[i].unit_index]
            if u.unit_type in ("table", "list", "code", "qa"):
                forbidden.append(
                    ForbiddenRange(
                        start=i,
                        end=i + 1,
                        forbidden_type="structure",
                    )
                )

    # 指代禁忌：代词到前一句
    for i, span in enumerate(sentences):
        if _CN_PRONOUNS.search(span.text) and i > 0:
            forbidden.append(
                ForbiddenRange(start=i - 1, end=i + 1, forbidden_type="reference")
            )

    # 括号平衡
    depth = 0
    open_idx = 0
    for i, span in enumerate(sentences):
        depth += span.text.count("(") + span.text.count("（")
        depth -= span.text.count(")") + span.text.count("）")
        if depth > 0 and open_idx == 0:
            open_idx = i
        if depth <= 0 and open_idx < i:
            forbidden.append(
                ForbiddenRange(start=open_idx, end=i + 1, forbidden_type="structure")
            )
            open_idx = 0
            depth = 0

    # 举例 / 定义
    for i, span in enumerate(sentences):
        if _EXAMPLE.search(span.text):
            end = min(n, i + 3)
            forbidden.append(
                ForbiddenRange(start=i, end=end, forbidden_type="example")
            )
        if _DEFINITION.search(span.text):
            forbidden.append(
                ForbiddenRange(start=i, end=i + 1, forbidden_type="definition")
            )

    return forbidden


def _in_forbidden(index: int, forbidden: List[ForbiddenRange]) -> bool:
    for fr in forbidden:
        if fr.start <= index < fr.end:
            return True
    return False


def _filter_cuts(
    cuts: Set[int],
    confidence: str,
    forbidden: List[ForbiddenRange],
) -> List[CutPoint]:
    out: List[CutPoint] = []
    for idx in sorted(cuts):
        if _in_forbidden(idx, forbidden):
            if confidence == "confirmed":
                out.append(CutPoint(sentence_index=idx, confidence="weak_A", reason="forbidden_downgrade"))
            continue
        out.append(CutPoint(sentence_index=idx, confidence=confidence, reason="boundary"))
    return out


def run_step2_boundaries(
    units: List[StructuralUnit],
    sentences: List[SentenceSpan],
    cfg: ChunkPipelineConfig,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
) -> BoundaryResult:
    """语义边界检测。"""
    cuts_b = _topic_tiling_cuts(
        sentences,
        window=max(1, cfg.topic_tiling_window),
        threshold=cfg.topic_tiling_threshold,
    )
    cuts_a: Set[int] = set()
    if cfg.use_embedding_boundary and embed_fn is not None and len(sentences) >= 3:
        try:
            cuts_a = _embedding_tiling_cuts(
                sentences,
                embed_fn=embed_fn,
                drop_threshold=cfg.embedding_drop_threshold,
            )
        except Exception:
            cuts_a = set()

    forbidden = _detect_forbidden_ranges(sentences, units)

    confirmed_idx = cuts_a & cuts_b
    weak_a_idx = cuts_a - cuts_b
    weak_b_idx = cuts_b - cuts_a

    return BoundaryResult(
        confirmed=_filter_cuts(confirmed_idx, "confirmed", forbidden),
        weak_a=_filter_cuts(weak_a_idx, "weak_A", forbidden),
        weak_b=_filter_cuts(weak_b_idx, "weak_B", forbidden),
        forbidden=forbidden,
    )
