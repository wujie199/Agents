"""Step4：父子层级构建。"""

import hashlib
import re
from typing import Dict, List, Tuple

from document.rag.config.chunk_pipeline import ChunkPipelineConfig
from document.rag.application.chunking.models import ScoredChunk
from document.rag.application.chunking.text_utils import keyword_set


def _parent_id(doc_id: str, content: str, index: int) -> str:
    h = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{doc_id}_parent_{index}_{h}"


def _child_id(doc_id: str, content: str, index: int) -> str:
    h = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{doc_id}_child_{index}_{h}"


def _summarize_parent(content: str, max_chars: int) -> str:
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if not lines:
        return content[:max_chars]
    head = lines[0][:80]
    body = content[: max_chars - len(head) - 3].replace("\n", " ")
    return f"{head} — {body}"[:max_chars]


def _split_into_child_parts(content: str, target_chars: int) -> List[str]:
    if len(content) <= target_chars:
        return [content]
    from document.rag.application.chunking.text_utils import split_sentences

    sentences = split_sentences(content)
    parts: List[str] = []
    buf: List[str] = []
    size = 0
    for sent in sentences:
        if buf and size + len(sent) > target_chars:
            parts.append("".join(buf))
            buf = [sent]
            size = len(sent)
        else:
            buf.append(sent)
            size += len(sent)
    if buf:
        parts.append("".join(buf))
    return parts or [content]


def _extract_entities(text: str) -> List[str]:
    tokens = keyword_set(text)
    return [t for t in tokens if len(t) >= 2][:32]


_DEFINITION_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9_\-]{2,20})(?:是指|定义为|是一种|指的是)([^。！？\n]{4,120})"
)


def _build_entity_index(chunks: List[ScoredChunk]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for ch in chunks:
        for match in _DEFINITION_RE.finditer(ch.content):
            index[match.group(1)] = match.group(2).strip()
    return index


def run_step4_hierarchy(
    chunks: List[ScoredChunk],
    doc_id: str,
    cfg: ChunkPipelineConfig,
) -> Tuple[List[ScoredChunk], List[ScoredChunk]]:
    """构建父子层级，返回 (retrieval_chunks, parent_chunks)。"""
    if not cfg.enable_parent_child or not chunks:
        for i, ch in enumerate(chunks):
            ch.chunk_role = "retrieval"
            ch.metadata.setdefault("chunk_role", "retrieval")
        return chunks, []

    entity_index = _build_entity_index(chunks)
    retrieval: List[ScoredChunk] = []
    parents: List[ScoredChunk] = []

    # FAQ / QA：每题独立父子对，不跨题合并
    if cfg.domain == "faq" or all(ch.unit_type == "qa" for ch in chunks):
        parent_idx = 0
        child_idx = 0
        for ch in chunks:
            pid = _parent_id(doc_id, ch.content, parent_idx)
            parent = ScoredChunk(
                content=ch.content,
                unit_type="qa",
                heading_path=ch.heading_path,
                position=parent_idx,
                density=ch.density,
                size=len(ch.content),
                metadata={"chunk_role": "parent", **ch.metadata},
                chunk_role="parent",
            )
            parent.metadata["chunk_id"] = pid
            cid = _child_id(doc_id, ch.content, child_idx)
            child = ScoredChunk(
                content=ch.content,
                unit_type="qa",
                heading_path=ch.heading_path,
                position=child_idx,
                density=ch.density,
                size=len(ch.content),
                parent_id=pid,
                metadata={"chunk_role": "retrieval", "parent_id": pid, **ch.metadata},
                chunk_role="retrieval",
            )
            child.metadata["chunk_id"] = cid
            if cfg.enable_contextual_prefix:
                child.contextualized_content = ch.content
            child.inherited_entities = {
                e: entity_index[e]
                for e in _extract_entities(child.content)
                if e in entity_index and entity_index[e] not in child.content
            }
            parent.child_ids = [cid]
            parents.append(parent)
            retrieval.append(child)
            parent_idx += 1
            child_idx += 1
        return retrieval, parents

    # 按 heading_path 分组；无路径则按 size 滑动窗口
    groups: Dict[str, List[ScoredChunk]] = {}
    for ch in chunks:
        key = ch.heading_path or f"__pos_{ch.position // 3}"
        groups.setdefault(key, []).append(ch)

    parent_idx = 0
    child_idx = 0
    for _, group in groups.items():
        group_text = "\n\n".join(c.content for c in group)
        if len(group_text) <= cfg.child_target_chars:
            ch = group[0]
            pid = _parent_id(doc_id, group_text, parent_idx)
            parent = ScoredChunk(
                content=group_text,
                unit_type=ch.unit_type,
                heading_path=ch.heading_path,
                position=parent_idx,
                density=ch.density,
                size=len(group_text),
                metadata={"chunk_role": "parent"},
                chunk_role="parent",
            )
            parent.metadata["chunk_id"] = pid
            parents.append(parent)

            cid = _child_id(doc_id, group_text, child_idx)
            child = ScoredChunk(
                content=group_text,
                unit_type=ch.unit_type,
                heading_path=ch.heading_path,
                position=child_idx,
                density=ch.density,
                size=len(group_text),
                parent_id=pid,
                metadata={"chunk_role": "retrieval", "parent_id": pid},
                chunk_role="retrieval",
            )
            child.metadata["chunk_id"] = cid
            if cfg.enable_contextual_prefix:
                prefix = _summarize_parent(group_text, cfg.contextual_prefix_max_chars)
                child.contextualized_content = f"{prefix}\n---\n{child.content}"
            child.inherited_entities = {
                e: entity_index[e]
                for e in _extract_entities(child.content)
                if e in entity_index and entity_index[e] not in child.content
            }
            retrieval.append(child)
            parent.child_ids = [cid]
            parent_idx += 1
            child_idx += 1
            continue

        pid = _parent_id(doc_id, group_text, parent_idx)
        parent = ScoredChunk(
            content=group_text,
            unit_type=group[0].unit_type,
            heading_path=group[0].heading_path,
            position=parent_idx,
            density=sum(c.density for c in group) / len(group),
            size=len(group_text),
            metadata={"chunk_role": "parent"},
            chunk_role="parent",
        )
        parent.metadata["chunk_id"] = pid
        parents.append(parent)
        child_ids: List[str] = []

        for part in _split_into_child_parts(group_text, cfg.child_target_chars):
            cid = _child_id(doc_id, part, child_idx)
            child = ScoredChunk(
                content=part,
                unit_type=group[0].unit_type,
                heading_path=group[0].heading_path,
                position=child_idx,
                density=_estimate_density_simple(part),
                size=len(part),
                parent_id=pid,
                metadata={"chunk_role": "retrieval", "parent_id": pid},
                chunk_role="retrieval",
            )
            child.metadata["chunk_id"] = cid
            if cfg.enable_contextual_prefix:
                prefix = _summarize_parent(group_text, cfg.contextual_prefix_max_chars)
                child.contextualized_content = f"{prefix}\n---\n{child.content}"
            child.inherited_entities = {
                e: entity_index[e]
                for e in _extract_entities(child.content)
                if e in entity_index and entity_index[e] not in child.content
            }
            retrieval.append(child)
            child_ids.append(cid)
            child_idx += 1

        parent.child_ids = child_ids
        parent_idx += 1

    return retrieval, parents


def _estimate_density_simple(text: str) -> float:
    tokens = keyword_set(text)
    if not text:
        return 0.0
    return min(1.0, len(tokens) / max(1, len(text) / 6))
