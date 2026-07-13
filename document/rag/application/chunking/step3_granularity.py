"""Step3：粒度自适应调整。"""

from typing import Dict, List, Set

from document.rag.config.chunk_pipeline import ChunkPipelineConfig
from document.rag.application.chunking.models import (
    BoundaryResult,
    ScoredChunk,
    SentenceSpan,
    StructuralUnit,
)
from document.rag.application.chunking.text_utils import keyword_set


def _estimate_density(text: str) -> float:
    tokens = keyword_set(text)
    if not text:
        return 0.0
    unique_ratio = len(tokens) / max(1, len(text) / 4)
    entity_hits = len([t for t in tokens if any(ch.isdigit() for ch in t) or len(t) >= 4])
    entity_ratio = entity_hits / max(1, len(tokens))
    return min(1.0, 0.5 * unique_ratio + 0.5 * entity_ratio)


def _target_for_density(density: float, cfg: ChunkPipelineConfig) -> Dict[str, int]:
    if cfg.domain in ("faq", "code"):
        return {"min": 0, "max": 10**9, "ideal": 10**9}
    if density >= cfg.density_high_threshold:
        return {
            "min": max(cfg.target_min, 100),
            "max": min(cfg.target_max, 400),
            "ideal": min(cfg.target_ideal, 300),
        }
    if density <= cfg.density_low_threshold:
        return {
            "min": max(cfg.target_min, 400),
            "max": min(cfg.max_chunk_size, 1500),
            "ideal": max(cfg.target_ideal, 600),
        }
    return {"min": cfg.target_min, "max": cfg.target_max, "ideal": cfg.target_ideal}


def _cut_index_sets(boundaries: BoundaryResult) -> Dict[str, Set[int]]:
    return {
        "confirmed": {c.sentence_index for c in boundaries.confirmed},
        "weak_a": {c.sentence_index for c in boundaries.weak_a},
        "weak_b": {c.sentence_index for c in boundaries.weak_b},
    }


def _forbidden_indices(boundaries: BoundaryResult) -> Set[int]:
    out: Set[int] = set()
    for fr in boundaries.forbidden:
        for i in range(fr.start, fr.end):
            out.add(i)
    return out


def _merge_unit_whole(
    unit: StructuralUnit,
    sentences: List[SentenceSpan],
    position: int,
) -> ScoredChunk:
    density = _estimate_density(unit.content)
    return ScoredChunk(
        content=unit.content,
        unit_type=unit.unit_type,
        heading_path=unit.heading_path,
        position=position,
        density=density,
        size=len(unit.content),
        metadata=dict(unit.metadata),
    )


def run_step3_granularity(
    units: List[StructuralUnit],
    sentences: List[SentenceSpan],
    boundaries: BoundaryResult,
    cfg: ChunkPipelineConfig,
) -> List[ScoredChunk]:
    """约束贪心切分。"""
    if not sentences:
        return []

    # FAQ / QA / code 整单元保留
    if cfg.domain == "faq" or all(u.unit_type in ("qa", "code") for u in units):
        return [
            _merge_unit_whole(u, sentences, idx)
            for idx, u in enumerate(units)
            if u.content.strip()
        ]
    if any(u.unit_type == "qa" for u in units):
        return [
            _merge_unit_whole(u, sentences, idx)
            for idx, u in enumerate(units)
            if u.content.strip()
        ]

    cuts = _cut_index_sets(boundaries)
    forbidden = _forbidden_indices(boundaries)

    chunks: List[ScoredChunk] = []
    buf: List[str] = []
    buf_units: List[int] = []
    buf_paths: List[str] = []
    buf_types: List[str] = []
    position = 0

    def flush() -> None:
        nonlocal position
        if not buf:
            return
        text = "".join(buf).strip()
        if not text:
            buf.clear()
            buf_units.clear()
            buf_paths.clear()
            buf_types.clear()
            return
        density = _estimate_density(text)
        unit_type = buf_types[0] if len(set(buf_types)) == 1 else "paragraph"
        heading_path = max(buf_paths, key=len) if buf_paths else ""
        meta: Dict = {}
        if buf_units:
            meta = dict(units[buf_units[0]].metadata)
        chunks.append(
            ScoredChunk(
                content=text,
                unit_type=unit_type,
                heading_path=heading_path,
                position=position,
                density=density,
                size=len(text),
                metadata=meta,
            )
        )
        position += 1
        buf.clear()
        buf_units.clear()
        buf_paths.clear()
        buf_types.clear()

    current_target = _target_for_density(0.5, cfg)

    i = 0
    while i < len(sentences):
        span = sentences[i]
        unit = units[span.unit_index]
        if unit.unit_type in ("table", "list", "code", "qa"):
            flush()
            chunks.append(_merge_unit_whole(unit, sentences, position))
            position += 1
            unit_idx = span.unit_index
            while i < len(sentences) and sentences[i].unit_index == unit_idx:
                i += 1
            continue

        buf.append(span.text)
        buf_units.append(span.unit_index)
        buf_paths.append(unit.heading_path)
        buf_types.append(unit.unit_type)
        cur_size = sum(len(x) for x in buf)
        current_target = _target_for_density(_estimate_density("".join(buf)), cfg)

        decision_index = i + 1
        must_cut = False
        prefer_cut = False

        if decision_index in forbidden:
            must_cut = False
        elif decision_index in cuts["confirmed"]:
            must_cut = True
        elif cur_size > current_target["max"]:
            must_cut = True
        elif (
            cur_size >= current_target["ideal"] * 0.85
            and decision_index in cuts["weak_a"]
        ):
            prefer_cut = True
        elif cur_size < current_target["min"]:
            must_cut = False
            prefer_cut = False

        if must_cut or prefer_cut:
            flush()
        i += 1

    flush()

    # 过小 chunk 合并
    merged: List[ScoredChunk] = []
    for ch in chunks:
        if merged and ch.size < cfg.min_chunk_size:
            prev = merged[-1]
            prev.content = prev.content + "\n" + ch.content
            prev.size = len(prev.content)
            prev.density = _estimate_density(prev.content)
        elif merged and merged[-1].size < cfg.min_chunk_size:
            prev = merged[-1]
            prev.content = prev.content + "\n" + ch.content
            prev.size = len(prev.content)
            prev.density = _estimate_density(prev.content)
        else:
            merged.append(ch)

    # 过大 chunk 二次切分（按 weak 切点或硬切）
    final: List[ScoredChunk] = []
    weak_points = sorted(cuts["weak_a"] | cuts["weak_b"])
    for ch in merged:
        if ch.size <= cfg.max_chunk_size:
            final.append(ch)
            continue
        from document.rag.application.chunking.text_utils import split_sentences

        sents = split_sentences(ch.content)
        sub_buf: List[str] = []
        sub_size = 0
        for sent in sents:
            if sub_buf and sub_size + len(sent) > cfg.max_chunk_size:
                final.append(
                    ScoredChunk(
                        content="".join(sub_buf),
                        unit_type=ch.unit_type,
                        heading_path=ch.heading_path,
                        position=ch.position,
                        density=ch.density,
                        size=sub_size,
                        metadata=dict(ch.metadata),
                    )
                )
                sub_buf = [sent]
                sub_size = len(sent)
            else:
                sub_buf.append(sent)
                sub_size += len(sent)
        if sub_buf:
            final.append(
                ScoredChunk(
                    content="".join(sub_buf),
                    unit_type=ch.unit_type,
                    heading_path=ch.heading_path,
                    position=ch.position,
                    density=ch.density,
                    size=sub_size,
                    metadata=dict(ch.metadata),
                )
            )
    return final
