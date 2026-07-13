"""Step5：Chunk 质量评分与过滤。"""

import re
from typing import List, Tuple

from document.rag.config.chunk_pipeline import ChunkPipelineConfig
from document.rag.application.chunking.models import RepairTask, ScoredChunk
from document.rag.application.chunking.text_utils import _CN_PRONOUNS, keyword_set


def _score_chunk(ch: ScoredChunk, cfg: ChunkPipelineConfig) -> Tuple[float, dict]:
    text = ch.content or ""
    tokens = keyword_set(text)
    char_len = max(1, len(text))

    # 信息密度
    info_density = min(1.0, len(tokens) / (char_len / 8))

    # 语义完整性
    completeness = 1.0
    if re.match(r"^(但是|因此|所以|然而|另外|而且)", text.strip()):
        completeness -= 0.35
    if re.search(r"(…|\.\.\.|：|:)\s*$", text.strip()):
        completeness -= 0.25
    completeness = max(0.0, completeness)

    # 实体丰富度
    entities = [t for t in tokens if len(t) >= 2]
    ch.entities = entities[:32]
    entity_richness = min(1.0, len(entities) / 8)

    # 可读性
    readability = 1.0
    if re.search(r"[\x00-\x08]", text):
        readability -= 0.5
    weird = sum(1 for c in text if ord(c) > 0xFFFF)
    if weird > 3:
        readability -= 0.3
    readability = max(0.0, readability)

    # 长度合理性
    approx_tokens = char_len // 2
    length_ok = 1.0
    if approx_tokens < cfg.min_chunk_size // 2:
        length_ok = 0.2
    elif approx_tokens > cfg.max_chunk_size:
        length_ok = 0.3

    dims = {
        "info_density": info_density,
        "completeness": completeness,
        "entity_richness": entity_richness,
        "readability": readability,
        "length_ok": length_ok,
    }
    score = (
        0.30 * info_density
        + 0.25 * completeness
        + 0.20 * entity_richness
        + 0.15 * readability
        + 0.10 * length_ok
    )
    return round(score, 4), dims


def run_step5_quality(
    chunks: List[ScoredChunk],
    cfg: ChunkPipelineConfig,
) -> Tuple[List[ScoredChunk], List[RepairTask]]:
    """质量评分、过滤、碎片合并、生成修复任务。"""
    threshold = cfg.quality_hard_filter
    if cfg.domain == "legal":
        threshold = min(threshold, 0.2)
    elif cfg.domain == "narrative":
        threshold = max(threshold, 0.35)

    scored: List[ScoredChunk] = []
    repair_tasks: List[RepairTask] = []

    for idx, ch in enumerate(chunks):
        score, dims = _score_chunk(ch, cfg)
        ch.score = score
        ch.dimension_scores = dims
        if score < threshold:
            continue
        if score < cfg.quality_soft_filter:
            ch.quality = "low"
        else:
            ch.quality = "high"
        scored.append(ch)

        if score < cfg.quality_repair_threshold:
            if dims["completeness"] < 0.6 and _CN_PRONOUNS.search(ch.content):
                repair_tasks.append(
                    RepairTask(task_type="reference", chunk_index=len(scored) - 1)
                )
            if dims["entity_richness"] < 0.4:
                repair_tasks.append(
                    RepairTask(task_type="entity", chunk_index=len(scored) - 1)
                )
            if dims["completeness"] < 0.6:
                repair_tasks.append(
                    RepairTask(task_type="context", chunk_index=len(scored) - 1)
                )
            if ch.unit_type in ("table", "list", "code") and dims["completeness"] < 0.7:
                repair_tasks.append(
                    RepairTask(task_type="structure", chunk_index=len(scored) - 1)
                )

    # 碎片合并：相邻 low quality 或过小 chunk
    merged: List[ScoredChunk] = []
    for ch in scored:
        if merged and (ch.size < cfg.min_chunk_size or ch.quality == "low"):
            prev = merged[-1]
            if prev.parent_id == ch.parent_id:
                prev.content = prev.content + "\n" + ch.content
                prev.size = len(prev.content)
                prev.score = max(prev.score, ch.score)
                continue
        merged.append(ch)

    return merged, repair_tasks
