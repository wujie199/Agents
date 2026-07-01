"""Article/chapter chunker for Chinese legal contracts."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

from core.ports.chunker import Chunk, ChunkerPort

_CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百千零\d]+章\s*(.*)$")
_ARTICLE_RE = re.compile(r"^第[一二三四五六七八九十百千零\d]+条\s*(.*)$")
_SECTION_RE = re.compile(r"^第[一二三四五六七八九十百千零\d]+[节款]\s*(.*)$")


class ArticleChunker:
    """Split contracts on 第X章 / 第X条 patterns with section_path metadata."""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        metadata = metadata or {}
        sections = self._parse_sections(text)
        chunks: List[Chunk] = []
        chunk_idx = 0

        for section in sections:
            header = section.get("header", "")
            section_path = section.get("section_path", "")
            content = section.get("content", "").strip()
            if not content and not header:
                continue

            full_content = f"{header}\n\n{content}".strip() if header else content
            if not full_content:
                continue

            if len(full_content) <= self._chunk_size:
                chunks.append(
                    self._make_chunk(
                        doc_id,
                        chunk_idx,
                        full_content,
                        metadata,
                        header=header,
                        section_path=section_path,
                    )
                )
                chunk_idx += 1
                continue

            from document.rag.application.indexing.chunker import RecursiveChunker

            sub_chunks = RecursiveChunker(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            ).chunk(full_content, doc_id, metadata)

            for sub in sub_chunks:
                chunks.append(
                    self._make_chunk(
                        doc_id,
                        chunk_idx,
                        sub.content,
                        metadata,
                        header=header,
                        section_path=section_path,
                    )
                )
                chunk_idx += 1

        return chunks

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

    def _parse_sections(self, text: str) -> List[Dict[str, str]]:
        lines = text.split("\n")
        sections: List[Dict[str, str]] = []
        current_chapter = ""
        current_header = ""
        current_path: List[str] = []
        current_content: List[str] = []

        def flush() -> None:
            nonlocal current_header, current_content
            body = "\n".join(current_content).strip()
            if current_header or body:
                path_parts = [p for p in current_path if p]
                sections.append(
                    {
                        "header": current_header,
                        "content": body,
                        "section_path": " > ".join(path_parts),
                    }
                )
            current_content = []

        for line in lines:
            stripped = line.strip()
            chapter_match = _CHAPTER_RE.match(stripped)
            article_match = _ARTICLE_RE.match(stripped)
            section_match = _SECTION_RE.match(stripped)

            if chapter_match:
                flush()
                current_chapter = stripped
                current_path = [current_chapter]
                current_header = current_chapter
                continue

            if article_match or section_match:
                flush()
                current_header = stripped
                path = [current_chapter] if current_chapter else []
                path.append(current_header)
                current_path = [p for p in path if p]
                continue

            if stripped:
                current_content.append(line)

        flush()

        if not sections and text.strip():
            sections.append({"header": "", "content": text.strip(), "section_path": ""})

        return sections

    def _make_chunk(
        self,
        doc_id: str,
        idx: int,
        content: str,
        metadata: Dict[str, Any],
        *,
        header: str,
        section_path: str,
    ) -> Chunk:
        chunk_meta = {
            **metadata,
            "strategy": "article",
            "header": header,
        }
        if section_path:
            chunk_meta["section_path"] = section_path
        return Chunk(
            chunk_id=self._generate_chunk_id(doc_id, idx, content),
            content=content,
            doc_id=doc_id,
            chunk_index=idx,
            metadata=chunk_meta,
            char_count=len(content),
        )

    def _generate_chunk_id(self, doc_id: str, idx: int, content: str) -> str:
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{doc_id}_chunk_{idx}_{content_hash}"
