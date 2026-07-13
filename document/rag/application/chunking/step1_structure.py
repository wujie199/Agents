"""Step1：结构识别与预分割。"""

import re
from typing import Any, Dict, List, Optional, Tuple

from document.rag.config.chunk_pipeline import ChunkPipelineConfig
from document.rag.application.chunking.models import StructuralUnit
from document.rag.shared.data_cleaner import _is_table_like_text, _merge_ocr_lines


_MD_HEADER = re.compile(r"^(#{1,6})\s+(.+)$")
_CODE_FENCE = re.compile(r"^```")
_LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")
_FAQ_Q = re.compile(r"^(\d+[、.．)]\s*(\*\*)?|Q[:：]\s*)", re.MULTILINE)
_FAQ_OCR_MARKDOWN = re.compile(r"(?:^|\n)\d+\.\*\*", re.MULTILINE)


def _detect_doc_format(metadata: Optional[Dict[str, Any]], content: str) -> str:
    meta = metadata or {}
    ext = str(meta.get("format") or meta.get("file_ext") or "").lower().lstrip(".")
    if ext in ("md", "markdown"):
        return "markdown"
    if ext in ("html", "htm"):
        return "html"
    if ext in ("doc", "docx"):
        return "word"
    if ext == "pdf":
        return "pdf"
    if content.lstrip().startswith("#"):
        return "markdown"
    if "<html" in content[:500].lower():
        return "html"
    return "plain"


def _is_faq_document(content: str, metadata: Optional[Dict[str, Any]]) -> bool:
    meta = metadata or {}
    if str(meta.get("chunk_domain") or "").lower() == "faq":
        return True
    source = str(meta.get("source_path") or "")
    if "常见问题" in source or "100问" in source or re.search(r"\d+问", source):
        return True
    sample = content[:6000]
    hits = len(_FAQ_Q.findall(sample))
    if hits >= 3:
        return True
    return len(_FAQ_OCR_MARKDOWN.findall(sample)) >= 3


def _effective_chunking_content(
    content: str,
    metadata: Optional[Dict[str, Any]],
) -> str:
    from document.rag.shared.ocr_ingest_text import rebuild_content_from_document_ir

    meta = metadata or {}
    if meta.get("ingest_backend") == "ocr_processor":
        ir_text = rebuild_content_from_document_ir(meta)
        if ir_text:
            return ir_text
    return content


def _split_faq_units(content: str, heading_path: str) -> List[StructuralUnit]:
    from document.rag.application.indexing.faq_chunker import split_faq_items

    items = split_faq_items(content)
    units: List[StructuralUnit] = []
    for idx, item in enumerate(items):
        path_parts = [p for p in (item.faq_category, item.faq_section) if p]
        unit_path = " > ".join(path_parts) if path_parts else heading_path
        units.append(
            StructuralUnit(
                unit_type="qa",
                content=item.content,
                heading_path=unit_path,
                position=idx,
                metadata={
                    "faq_number": item.faq_number,
                    "faq_category": item.faq_category,
                    "faq_section": item.faq_section,
                },
            )
        )
    return units


def _parse_markdown_units(content: str) -> List[StructuralUnit]:
    lines = content.splitlines()
    units: List[StructuralUnit] = []
    heading_stack: List[Tuple[int, str]] = []
    buffer: List[str] = []
    in_code = False
    position = 0

    def flush_paragraph() -> None:
        nonlocal position
        text = "\n".join(buffer).strip()
        buffer.clear()
        if not text:
            return
        path = " > ".join(h for _, h in heading_stack)
        units.append(
            StructuralUnit(
                unit_type="paragraph",
                content=text,
                heading_path=path,
                position=position,
            )
        )
        position += 1

    def flush_code(code_lines: List[str]) -> None:
        nonlocal position
        text = "\n".join(code_lines).strip()
        if not text:
            return
        path = " > ".join(h for _, h in heading_stack)
        units.append(
            StructuralUnit(
                unit_type="code",
                content=text,
                heading_path=path,
                position=position,
            )
        )
        position += 1

    code_buf: List[str] = []
    list_buf: List[str] = []

    def flush_list() -> None:
        nonlocal position
        if not list_buf:
            return
        text = "\n".join(list_buf).strip()
        list_buf.clear()
        path = " > ".join(h for _, h in heading_stack)
        units.append(
            StructuralUnit(
                unit_type="list",
                content=text,
                heading_path=path,
                position=position,
            )
        )
        position += 1

    for line in lines:
        if _CODE_FENCE.match(line.strip()):
            if in_code:
                code_buf.append(line)
                flush_code(code_buf)
                code_buf = []
                in_code = False
            else:
                flush_paragraph()
                flush_list()
                in_code = True
                code_buf = [line]
            continue
        if in_code:
            code_buf.append(line)
            continue

        header_match = _MD_HEADER.match(line)
        if header_match:
            flush_paragraph()
            flush_list()
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            path = " > ".join(h for _, h in heading_stack)
            units.append(
                StructuralUnit(
                    unit_type="heading",
                    content=title,
                    heading_path=path,
                    position=position,
                )
            )
            position += 1
            continue

        if _LIST_ITEM.match(line):
            flush_paragraph()
            list_buf.append(line)
            continue
        if list_buf:
            flush_list()

        buffer.append(line)

    flush_paragraph()
    flush_list()
    if in_code and code_buf:
        flush_code(code_buf)

    # 二次扫描：把连续 table-like 段落提升为 table 单元
    merged: List[StructuralUnit] = []
    i = 0
    while i < len(units):
        u = units[i]
        if u.unit_type == "paragraph" and _is_table_like_text(u.content):
            merged.append(
                StructuralUnit(
                    unit_type="table",
                    content=u.content,
                    heading_path=u.heading_path,
                    position=u.position,
                )
            )
        else:
            merged.append(u)
        i += 1
    return merged


def _split_plain_units(content: str) -> List[StructuralUnit]:
    text = _merge_ocr_lines(content) if "\n" in content else content
    blocks = re.split(r"\n\s*\n", text)
    units: List[StructuralUnit] = []
    for idx, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue
        unit_type = "paragraph"
        if _is_table_like_text(block):
            unit_type = "table"
        elif _CODE_FENCE.search(block):
            unit_type = "code"
        elif _LIST_ITEM.search(block):
            unit_type = "list"
        units.append(
            StructuralUnit(
                unit_type=unit_type,
                content=block,
                heading_path="",
                position=idx,
            )
        )
    return units


def _split_long_paragraph(unit: StructuralUnit, max_chars: int) -> List[StructuralUnit]:
    if unit.unit_type in ("table", "list", "code", "qa", "heading"):
        return [unit]
    if len(unit.content) <= max_chars:
        return [unit]
    from document.rag.application.chunking.text_utils import split_sentences

    sentences = split_sentences(unit.content)
    parts: List[StructuralUnit] = []
    buf: List[str] = []
    length = 0
    pos = unit.position
    for sent in sentences:
        if buf and length + len(sent) > max_chars:
            parts.append(
                StructuralUnit(
                    unit_type="paragraph",
                    content="".join(buf),
                    heading_path=unit.heading_path,
                    position=pos,
                )
            )
            pos += 1
            buf = [sent]
            length = len(sent)
        else:
            buf.append(sent)
            length += len(sent)
    if buf:
        parts.append(
            StructuralUnit(
                unit_type="paragraph",
                content="".join(buf),
                heading_path=unit.heading_path,
                position=pos,
            )
        )
    return parts


def _split_table_by_row_groups(unit: StructuralUnit, max_chars: int) -> List[StructuralUnit]:
    if len(unit.content) <= max_chars:
        return [unit]
    lines = [ln for ln in unit.content.splitlines() if ln.strip()]
    if len(lines) < 3:
        return [unit]
    header = lines[0]
    rows = lines[1:]
    groups: List[StructuralUnit] = []
    buf = [header]
    size = len(header)
    pos = unit.position
    for row in rows:
        if buf and size + len(row) > max_chars:
            groups.append(
                StructuralUnit(
                    unit_type="table",
                    content="\n".join(buf),
                    heading_path=unit.heading_path,
                    position=pos,
                    metadata={"table_partial": True},
                )
            )
            pos += 1
            buf = [header, row]
            size = len(header) + len(row)
        else:
            buf.append(row)
            size += len(row)
    if len(buf) > 1:
        groups.append(
            StructuralUnit(
                unit_type="table",
                content="\n".join(buf),
                heading_path=unit.heading_path,
                position=pos,
                metadata={"table_partial": True},
            )
        )
    return groups or [unit]


def run_step1_structure(
    content: str,
    cfg: ChunkPipelineConfig,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[StructuralUnit]:
    """结构识别与预分割。"""
    meta = metadata or {}
    effective_content = _effective_chunking_content(content, meta)

    from document.rag.application.chunking.ir_to_structure import (
        get_document_ir_from_metadata,
        units_from_document_ir,
    )

    if cfg.preserve_faq_pairs and _is_faq_document(effective_content, meta):
        units = _split_faq_units(effective_content, heading_path="FAQ")
        max_unit = cfg.target_max or cfg.max_chunk_size
        expanded: List[StructuralUnit] = []
        for unit in units:
            if unit.unit_type == "table" and cfg.preserve_tables:
                expanded.extend(_split_table_by_row_groups(unit, max_unit))
            elif unit.unit_type == "paragraph":
                expanded.extend(_split_long_paragraph(unit, max_unit))
            else:
                expanded.append(unit)
        for unit in expanded:
            if len(unit.content) < cfg.min_unit_size:
                unit.is_fragment = True
        return expanded

    document_ir = get_document_ir_from_metadata(meta)
    if document_ir and meta.get("ingest_backend") == "ocr_processor":
        units = units_from_document_ir(
            document_ir,
            default_heading=str(meta.get("doc_title") or ""),
        )
        if units:
            max_unit = cfg.target_max or cfg.max_chunk_size
            expanded: List[StructuralUnit] = []
            for unit in units:
                if unit.unit_type == "table" and cfg.preserve_tables:
                    expanded.extend(_split_table_by_row_groups(unit, max_unit))
                elif unit.unit_type == "paragraph":
                    expanded.extend(_split_long_paragraph(unit, max_unit))
                else:
                    expanded.append(unit)
            for unit in expanded:
                if len(unit.content) < cfg.min_unit_size:
                    unit.is_fragment = True
            return expanded

    doc_format = _detect_doc_format(meta, effective_content)

    if doc_format == "markdown":
        units = _parse_markdown_units(effective_content)
    else:
        units = _split_plain_units(effective_content)

    max_unit = cfg.target_max or cfg.max_chunk_size
    expanded: List[StructuralUnit] = []
    for unit in units:
        if unit.unit_type == "table" and cfg.preserve_tables:
            expanded.extend(_split_table_by_row_groups(unit, max_unit))
        elif unit.unit_type == "paragraph":
            expanded.extend(_split_long_paragraph(unit, max_unit))
        else:
            expanded.append(unit)

    for unit in expanded:
        if len(unit.content) < cfg.min_unit_size:
            unit.is_fragment = True

    return expanded
