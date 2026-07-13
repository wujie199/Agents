"""FAQ 专用切块：按题号将「一问一答」切为独立 chunk。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.ports.chunker import Chunk

from document.rag.shared.text_chunker import split_text_into_chunks

_PAGE_PLACEHOLDER = "__FAQ_PAGE_{n}__"
_TITLE_PLACEHOLDER = "__FAQ_DOC_TITLE__"
# OCR Paddle 输出：1.**题干?** -答案
_OCR_FAQ_START = re.compile(r"(?:^|\n)(\d{1,3})\.\*\*", re.MULTILINE)
_FAQ_START = re.compile(
    r"(?:(?<=\n)|(?<=^)|(?<=[^\d]))"
    r"(\d{1,3})"
    r"[\.、]?\s*"
    r"(?=[\u4e00-\u9fff\"「])",
    re.MULTILINE,
)
_PAGE_MARKER = re.compile(r"={2,}\s*第\s*(\d{1,3})\s*页\s*={0,3}", re.MULTILINE)
_DOC_TITLE_100 = re.compile(r"#?\s*扫地机器人100问")
_CATEGORY_RE = re.compile(r"[一二三四五六七八九十百]+[\u4e00-\u9fff]+类")
_SECTION_WITH_SUFFIX = re.compile(
    r"^[\u4e00-\u9fff]{1,2}(?:系统|指南|技巧|规划)$"
)
_QUESTION_HINT = re.compile(
    r"(什么是|为什么|如何|怎么|能否|可以|是否|有没有|哪些|什么|怎样|有必要|适合|支持|需要|会|能|应|该|区别|影响|提高|解决|处理|导致|造成|出现|使用|购买|选购|安装|设置|清洁|保养|避免|防止|识别|检测|定位|导航|建图|充电|回充|拖地|清扫|耗材|电池|地图|传感器|固件|语音|故障|报错|报警|异常|漏扫|卡住|迷路|被困|不工作|不能|不会|不知道|机器人|扫地|拖地|水箱|尘盒|充电座|集尘|自动|手动|遥控|安全|隐私|数据|WiFi|APP|LDS|SLAM|VSLAM|dToF|ToF)",
    re.IGNORECASE,
)


@dataclass
class FaqItem:
    content: str
    faq_number: str
    faq_category: Optional[str] = None
    faq_section: Optional[str] = None


def is_section_title(line: str) -> bool:
    """判断是否为 FAQ 章节/小节标题（非问句、非题号行）。"""
    line = line.strip()
    if not line or len(line) > 16:
        return False
    if line[0].isdigit() or "？" in line or "?" in line:
        return False
    if _CATEGORY_RE.fullmatch(line):
        return True
    if _SECTION_WITH_SUFFIX.match(line):
        return True
    if re.match(r"^[\u4e00-\u9fff]{2,8}(?:与[\u4e00-\u9fff]{2,8})+$", line) and len(line) <= 12:
        return True
    return False


def extract_trailing_section(text: str) -> Tuple[str, Optional[str]]:
    """从答案末尾剥离下一节标题（如「清洁系统」），返回 (净文本, 被剥离标题)。"""
    stripped: Optional[str] = None
    lines = text.rstrip().split("\n")
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if is_section_title(last):
            stripped = last
            lines.pop()
        else:
            break
    text = "\n".join(lines).strip()

    glued = re.search(
        r"([\u4e00-\u9fff]{1,2}(?:系统|指南|技巧|规划)|"
        r"[\u4e00-\u9fff]{2,8}与[\u4e00-\u9fff]{2,8})$",
        text,
    )
    if glued and is_section_title(glued.group(1)):
        text = text[: glued.start(1)].rstrip()
        stripped = glued.group(1)
    return text, stripped


def headers_before(text: str, end_pos: int) -> Tuple[Optional[str], Optional[str]]:
    """扫描题号之前的全文，取最近的大类与小节标题。"""
    prefix = text[:end_pos]
    categories = _CATEGORY_RE.findall(prefix)
    faq_category = categories[-1] if categories else None

    faq_section: Optional[str] = None
    for line in prefix.split("\n"):
        candidate = line.strip()
        if is_section_title(candidate) and not _CATEGORY_RE.fullmatch(candidate):
            faq_section = candidate
            continue
        for token in re.split(r"\s+", candidate):
            if is_section_title(token) and not _CATEGORY_RE.fullmatch(token):
                faq_section = token
        glued = re.search(
            r"([\u4e00-\u9fff]{1,2}(?:系统|指南|技巧|规划)|"
            r"[\u4e00-\u9fff]{2,8}与[\u4e00-\u9fff]{2,8})$",
            candidate,
        )
        if glued and is_section_title(glued.group(1)):
            faq_section = glued.group(1)
    return faq_category, faq_section


def sanitize_faq_content(content: str) -> str:
    """去掉清洗/切块产生的 -？ 等 artifact。"""
    s = re.sub(r"\n-\s*[？?]\s*\n?", "\n", content)
    s = re.sub(r"(?<=\n)-\s*[？?]\s*", "", s)
    return s.strip()


def normalize_faq_text(text: str) -> str:
    """在 OCR 粘连的题号/章节前插入换行，并保护页码/文档标题。"""
    if not text:
        return ""
    s = text.replace("\ufeff", "").replace("\ufffd", "")
    s = _DOC_TITLE_100.sub(_TITLE_PLACEHOLDER, s)
    s = _PAGE_MARKER.sub(lambda m: f"\n{_PAGE_PLACEHOLDER.format(n=m.group(1))}\n", s)
    s = re.sub(r"第\s*(\d{1,2})\s*页", lambda m: f"\n{_PAGE_PLACEHOLDER.format(n=m.group(1))}\n", s)
    # OCR Markdown 题号：确保每题独占行首
    s = re.sub(r"([^\n\d])(\d{1,3}\.\*\*)", r"\1\n\2", s)
    s = re.sub(r"(\D)([一二三四五六七八九十百]+[\u4e00-\u9fff]+类)", r"\1\n\2", s)
    s = re.sub(
        r"([一二三四五六七八九十百]+[\u4e00-\u9fff]+类)\s*"
        r"([\u4e00-\u9fff]{2,8}(?:与[\u4e00-\u9fff]{2,8})?)"
        r"(?=\s*\d{1,3}(?:\.\*\*|[\.、]))",
        r"\1\n\2\n",
        s,
    )
    s = re.sub(
        r"([\u4e00-\u9fff])"
        r"([\u4e00-\u9fff]{1,2}(?:系统|指南|技巧|规划))"
        r"(?=\d{1,3}(?:\.\*\*|[\.、]))",
        r"\1\n\2\n",
        s,
    )
    s = re.sub(
        r"([\u4e00-\u9fff。！？;；])(\d{1,3})(?=\.\*\*)",
        r"\1\n\2",
        s,
    )
    s = re.sub(
        r"([\u4e00-\u9fff。！？])(\d{1,3})(?=\s+[A-Za-z\u4e00-\u9fff\"「])",
        r"\1\n\2",
        s,
    )
    s = re.sub(r"(\D)(\d{1,3})(?=[\u4e00-\u9fff\"「])", r"\1\n\2", s)
    s = re.sub(r"__FAQ_PAGE_\d+__", "\n", s)
    s = re.sub(_TITLE_PLACEHOLDER, "#扫地机器人100问", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _faq_start_matches(text: str) -> List[re.Match[str]]:
    """优先 OCR Markdown 题号，否则回退 plain OCR。"""
    ocr = list(_OCR_FAQ_START.finditer(text))
    if len(ocr) >= 3:
        return ocr
    plain = list(_FAQ_START.finditer(text))
    return plain if plain else ocr


def _is_false_faq_match(num: str, block: str) -> bool:
    """过滤页码/标题误匹配。"""
    if num == "100" and ("问" in block[:24] or _TITLE_PLACEHOLDER in block):
        return True
    if num in {"1", "2", "3", "4", "5", "6", "7", "8"} and block.strip().startswith(
        ("页", "===", "第 ")
    ):
        return True
    return False


def _format_ocr_markdown_faq(
    num: str,
    rest: str,
    *,
    faq_category: Optional[str] = None,
    faq_section: Optional[str] = None,
) -> FaqItem:
    rest = rest.strip()
    question = ""
    answer = ""
    qm = re.match(r"^(.+?)\*\*(.*)$", rest, re.DOTALL)
    if qm:
        question = qm.group(1).strip()
        answer = qm.group(2).strip()
    else:
        parts = re.split(
            r"(?<=[？?])[ \-—–]*|(?:[ \-—–](?=[\u4e00-\u9fff]))",
            rest,
            maxsplit=1,
        )
        if len(parts) == 2:
            question, answer = parts[0].strip(), parts[1].strip()
        else:
            question = rest

    question = re.sub(r"^\*\*|\*\*$", "", question).strip()
    answer = re.sub(r"^[\*\?\s\-—–]+", "", answer)
    answer = re.sub(r"^\*\*\s*", "", answer)
    if question and not question.endswith(("？", "?")) and _QUESTION_HINT.search(question):
        question += "？"

    if answer:
        answer, _trailing = extract_trailing_section(answer)
        content = sanitize_faq_content(f"{num}. {question}\n{answer}".strip())
    else:
        content = sanitize_faq_content(f"{num}. {question}".strip())

    return FaqItem(
        content=content,
        faq_number=num,
        faq_category=faq_category,
        faq_section=faq_section,
    )


def format_faq_block(
    block: str,
    *,
    faq_category: Optional[str] = None,
    faq_section: Optional[str] = None,
) -> FaqItem:
    """将单题块格式化为「题号. 题干\\n答案」，章节标题进 metadata。"""
    block = block.strip()
    ocr_m = re.match(r"^(\d{1,3})\.\*\*(.+)$", block, re.DOTALL)
    if ocr_m:
        return _format_ocr_markdown_faq(
            ocr_m.group(1),
            ocr_m.group(2),
            faq_category=faq_category,
            faq_section=faq_section,
        )

    m = re.match(r"^(\d{1,3})[\.、]?\s*(.+)$", block, re.DOTALL)
    if not m:
        return FaqItem(content=block, faq_number="", faq_category=faq_category, faq_section=faq_section)

    num, body = m.group(1), m.group(2).strip()
    parts = re.split(
        r"(?<=[？?])[ \-—]*|(?:[ \-——](?=[\u4e00-\u9fff]))",
        body,
        maxsplit=1,
    )
    if len(parts) == 2:
        question, answer = parts[0].strip(), parts[1].strip()
        if question and not question.endswith(("？", "?")):
            if _QUESTION_HINT.search(question):
                question += "？"
        answer, _trailing = extract_trailing_section(answer)
        content = sanitize_faq_content(f"{num}. {question}\n{answer}".strip())
        return FaqItem(
            content=content,
            faq_number=num,
            faq_category=faq_category,
            faq_section=faq_section,
        )
    body, _trailing = extract_trailing_section(body)
    return FaqItem(
        content=sanitize_faq_content(f"{num}. {body}"),
        faq_number=num,
        faq_category=faq_category,
        faq_section=faq_section,
    )


def split_faq_items(text: str, *, min_chars: int = 8) -> List[FaqItem]:
    """按题号切分 FAQ 条目；条目不足时返回空列表（由调用方 fallback）。"""
    normalized = normalize_faq_text(text)
    matches = _faq_start_matches(normalized)
    if len(matches) < 1:
        return []

    items: List[FaqItem] = []
    for i, match in enumerate(matches):
        start = match.start(1)
        end = matches[i + 1].start(1) if i + 1 < len(matches) else len(normalized)
        block = normalized[start:end].strip()
        num = match.group(1)
        if len(block) < min_chars or _is_false_faq_match(num, block):
            continue
        faq_category, faq_section = headers_before(normalized, start)
        items.append(
            format_faq_block(
                block,
                faq_category=faq_category,
                faq_section=faq_section,
            )
        )
    return items


class FaqChunker:
    """按 FAQ 题号切块；单题过长时在该题内部递归二次切分。"""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 0,
        min_item_chars: int = 8,
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_item_chars = min_item_chars

    def _text_chunks(self, text: str) -> List[str]:
        return split_text_into_chunks(
            text,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        metadata = metadata or {}
        items = split_faq_items(text, min_chars=self._min_item_chars)
        if not items:
            parts = self._text_chunks(text)
            return [
                Chunk(
                    chunk_id=self._generate_chunk_id(doc_id, idx, part),
                    content=part,
                    doc_id=doc_id,
                    chunk_index=idx,
                    metadata={**metadata, "strategy": "faq_fallback_split"},
                    char_count=len(part),
                )
                for idx, part in enumerate(parts)
            ] or []

        result: List[Chunk] = []
        chunk_idx = 0
        for item in items:
            sub_chunks = self._split_long_item(item.content, doc_id, metadata, item.faq_number)
            for sub in sub_chunks:
                chunk_id = self._generate_chunk_id(doc_id, chunk_idx, sub)
                chunk_meta = {
                    **metadata,
                    "strategy": "faq",
                    "faq_number": item.faq_number,
                }
                if item.faq_category:
                    chunk_meta["faq_category"] = item.faq_category
                if item.faq_section:
                    chunk_meta["faq_section"] = item.faq_section
                result.append(
                    Chunk(
                        chunk_id=chunk_id,
                        content=sub,
                        doc_id=doc_id,
                        chunk_index=chunk_idx,
                        metadata=chunk_meta,
                        char_count=len(sub),
                    )
                )
                chunk_idx += 1
        return result

    def chunk_batch(
        self,
        texts: List[str],
        doc_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[List[Chunk]]:
        return [
            self.chunk(text, doc_id, metadata)
            for text, doc_id in zip(texts, doc_ids)
        ]

    def _split_long_item(
        self,
        item: str,
        doc_id: str,
        metadata: Dict[str, Any],
        faq_num: Optional[str],
    ) -> List[str]:
        if len(item) <= self._chunk_size:
            return [item]

        inner_parts = self._text_chunks(item)
        parts: List[str] = []
        for i, part in enumerate(inner_parts):
            parts.append(f"{faq_num}. (续) {part}" if i > 0 else part)
        return parts or [item]

    def _generate_chunk_id(self, doc_id: str, idx: int, content: str) -> str:
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{doc_id}_chunk_{idx}_{content_hash}"
