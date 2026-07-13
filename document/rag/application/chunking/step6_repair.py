"""Step6：上下文修复。"""

import re
from typing import Dict, List

from document.rag.config.chunk_pipeline import ChunkPipelineConfig
from document.rag.application.chunking.models import RepairTask, ScoredChunk
from document.rag.application.chunking.text_utils import _CN_PRONOUNS, keyword_set


def _build_entity_definition_index(
    original_content: str,
    chunks: List[ScoredChunk],
) -> Dict[str, str]:
    index: Dict[str, str] = {}
    corpus = original_content + "\n" + "\n".join(c.content for c in chunks)
    for match in re.finditer(
        r"([\u4e00-\u9fffA-Za-z0-9_\-]{2,20})(?:是指|定义为|是一种|指的是)([^。！？\n]{4,200})",
        corpus,
    ):
        index[match.group(1)] = match.group(2).strip()
    for match in re.finditer(
        r"([\u4e00-\u9fffA-Za-z0-9_\-]{2,20})[（(]([^）)]+)[）)]",
        corpus,
    ):
        index.setdefault(match.group(1), match.group(2).strip())
    return index


def _find_antecedent_snippet(content: str, chunks: List[ScoredChunk], idx: int) -> str:
    if idx > 0:
        prev = chunks[idx - 1].content.strip()
        if prev:
            return prev[-80:]
    for j in range(idx - 1, -1, -1):
        if chunks[j].heading_path == chunks[idx].heading_path:
            return chunks[j].content[-120:]
    return ""


def run_step6_repair(
    chunks: List[ScoredChunk],
    repair_tasks: List[RepairTask],
    original_content: str,
    cfg: ChunkPipelineConfig,
) -> List[ScoredChunk]:
    """指代/实体/结构修复。"""
    if not cfg.enable_context_repair or not chunks:
        return chunks

    entity_index = _build_entity_definition_index(original_content, chunks)
    task_map: Dict[int, List[RepairTask]] = {}
    for task in repair_tasks:
        task_map.setdefault(task.chunk_index, []).append(task)

    for idx, ch in enumerate(chunks):
        tasks = task_map.get(idx, [])
        if not tasks:
            continue

        for task in tasks:
            if task.task_type == "reference" and _CN_PRONOUNS.search(ch.content):
                snippet = _find_antecedent_snippet(ch.content, chunks, idx)
                if snippet and len(snippet) < 50:
                    ch.content = f"[上下文: {snippet}]\n{ch.content}"
                    ch.size = len(ch.content)
                elif snippet:
                    ch.metadata.setdefault("reference_context", snippet)

            elif task.task_type == "entity":
                used = _extract_used_entities(ch.content)
                defs: Dict[str, str] = dict(ch.inherited_entities)
                for ent in used:
                    if ent in entity_index and entity_index[ent] not in ch.content:
                        definition = entity_index[ent]
                        if len(definition) <= cfg.entity_definition_max_chars:
                            ch.content += f"\n[定义: {ent} — {definition}]"
                        else:
                            defs[ent] = definition
                ch.inherited_entities = defs
                ch.size = len(ch.content)

            elif task.task_type == "context":
                if not ch.content.strip().endswith(("。", "！", "？", ".", "!", "?")):
                    nxt = chunks[idx + 1].content[:60] if idx + 1 < len(chunks) else ""
                    if nxt:
                        ch.metadata["tail_context_hint"] = nxt

            elif task.task_type == "structure":
                if ch.unit_type == "table" and "|" not in ch.content and "\t" not in ch.content:
                    # 从原文回填 table-like 块
                    block = _extract_block_from_original(original_content, ch.content[:40])
                    if block:
                        ch.content = block
                        ch.size = len(ch.content)
                if ch.unit_type == "code" and "```" not in ch.content:
                    block = _extract_codeblock_from_original(original_content, ch.content[:40])
                    if block:
                        ch.content = block
                        ch.size = len(ch.content)

        if ch.contextualized_content and not ch.contextualized_content.endswith(ch.content):
            prefix = ch.contextualized_content.split("\n---\n", 1)[0]
            ch.contextualized_content = f"{prefix}\n---\n{ch.content}"

    return chunks


def _extract_used_entities(text: str) -> List[str]:
    return [t for t in keyword_set(text) if len(t) >= 2][:16]


def _extract_block_from_original(original: str, hint: str) -> str:
    if not hint or hint not in original:
        return ""
    start = original.find(hint)
    if start < 0:
        return ""
    snippet = original[start : start + 2000]
    lines = snippet.splitlines()
    if len(lines) < 2:
        return snippet.strip()
    return "\n".join(lines[: min(20, len(lines))]).strip()


def _extract_codeblock_from_original(original: str, hint: str) -> str:
    if not hint:
        return ""
    pos = original.find(hint)
    if pos < 0:
        return ""
    window = original[max(0, pos - 200) : pos + 1200]
    match = re.search(r"```[\s\S]*?```", window)
    return match.group(0) if match else ""
