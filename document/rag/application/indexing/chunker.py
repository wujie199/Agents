import re
import hashlib
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from core.ports.chunker import ChunkerPort, Chunk, ChunkStrategy


class RecursiveChunker:
    """Recursive text chunker with paragraph/sentence/word separators."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or [
            "\n\n",
            "\n",
            "。",
            "，",
            " ",
            ".",
            "!",
            "?",
            " ",
            "",
        ]
        self._keep_separator = keep_separator
    
    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        metadata = metadata or {}
        
        splits = self._split_text(text)
        
        chunks = self._merge_splits(splits)
        
        result = []
        for idx, chunk_content in enumerate(chunks):
            chunk_id = self._generate_chunk_id(doc_id, idx, chunk_content)
            
            result.append(Chunk(
                chunk_id=chunk_id,
                content=chunk_content,
                doc_id=doc_id,
                chunk_index=idx,
                metadata={**metadata, "strategy": "recursive"},
                char_count=len(chunk_content),
            ))
        
        return result
    
    def chunk_batch(
        self,
        texts: List[str],
        doc_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[List[Chunk]]:
        return [
            self.chunk(text, doc_id, metadata)
            for text, doc_id in zip(texts, doc_ids)
        ]
    
    def _split_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        for separator in self._separators:
            if separator == "":
                return list(text)
            
            if separator in text:
                splits = text.split(separator)
                
                if self._keep_separator:
                    result = []
                    for i, split in enumerate(splits):
                        if i < len(splits) - 1:
                            result.append(split + separator)
                        else:
                            result.append(split)
                    splits = result
                
                final_splits = []
                for split in splits:
                    if split:
                        if len(split) > self._chunk_size:
                            final_splits.extend(self._split_text(split))
                        else:
                            final_splits.append(split)
                
                return final_splits
        
        return [text]
    
    def _merge_splits(self, splits: List[str]) -> List[str]:
        if not splits:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split)
            
            if current_length + split_length <= self._chunk_size:
                current_chunk.append(split)
                current_length += split_length
            else:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                
                if self._chunk_overlap > 0 and current_chunk:
                    overlap_text = "".join(current_chunk)[-self._chunk_overlap:]
                    current_chunk = [overlap_text, split]
                    current_length = len(overlap_text) + split_length
                else:
                    current_chunk = [split]
                    current_length = split_length
        
        if current_chunk:
            chunks.append("".join(current_chunk))
        
        return chunks
    
    def _generate_chunk_id(self, doc_id: str, idx: int, content: str) -> str:
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{doc_id}_chunk_{idx}_{content_hash}"


class MarkdownChunker:
    """Chunk markdown by headers, then recursive split within sections."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
    
    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        metadata = metadata or {}
        
        sections = self._parse_markdown(text)
        
        chunks = []
        chunk_idx = 0
        
        for section in sections:
            content = section["content"]
            header = section.get("header", "")
            
            full_content = f"{header}\n\n{content}" if header else content
            
            if len(full_content) <= self._chunk_size:
                chunk_id = self._generate_chunk_id(doc_id, chunk_idx, full_content)
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    content=full_content,
                    doc_id=doc_id,
                    chunk_index=chunk_idx,
                    metadata={
                        **metadata,
                        "strategy": "markdown",
                        "header": header,
                        "level": section.get("level", 0),
                    },
                ))
                chunk_idx += 1
            else:
                recursive_chunker = RecursiveChunker(
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._chunk_overlap
                )
                sub_chunks = recursive_chunker.chunk(full_content, doc_id, metadata)
                
                for sub in sub_chunks:
                    chunks.append(Chunk(
                        chunk_id=self._generate_chunk_id(doc_id, chunk_idx, sub.content),
                        content=sub.content,
                        doc_id=doc_id,
                        chunk_index=chunk_idx,
                        metadata={**sub.metadata, "header": header},
                    ))
                    chunk_idx += 1
        
        return chunks
    
    def chunk_batch(
        self,
        texts: List[str],
        doc_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[List[Chunk]]:
        return [
            self.chunk(text, doc_id, metadata)
            for text, doc_id in zip(texts, doc_ids)
        ]
    
    def _parse_markdown(self, text: str) -> List[Dict[str, Any]]:
        lines = text.split('\n')
        sections = []
        current_section = {"header": "", "content": "", "level": 0}
        
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                if current_section["content"].strip():
                    sections.append(current_section)
                
                level = len(header_match.group(1))
                header = header_match.group(2)
                current_section = {"header": header, "content": "", "level": level}
            else:
                current_section["content"] += line + '\n'
        
        if current_section["content"].strip():
            sections.append(current_section)
        
        return sections
    
    def _generate_chunk_id(self, doc_id: str, idx: int, content: str) -> str:
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{doc_id}_chunk_{idx}_{content_hash}"


class SemanticChunker:
    """Semantic chunker (MVP: delegates to recursive chunking)."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self._inner = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        chunks = self._inner.chunk(text, doc_id, metadata)
        for c in chunks:
            c.metadata["strategy"] = "semantic"
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


def create_chunker(
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    **kwargs,
) -> ChunkerPort:
    """Chunker factory."""
    if strategy == ChunkStrategy.FAQ:
        from document.rag.application.indexing.faq_chunker import FaqChunker

        return FaqChunker(**kwargs)
    if strategy == ChunkStrategy.RECURSIVE:
        return RecursiveChunker(**kwargs)
    if strategy == ChunkStrategy.MARKDOWN:
        return MarkdownChunker(**kwargs)
    if strategy == ChunkStrategy.SEMANTIC:
        return SemanticChunker(**kwargs)
    return RecursiveChunker(**kwargs)


def parse_chunk_strategy(name: str) -> ChunkStrategy:
    key = (name or "recursive").lower()
    mapping = {
        "recursive": ChunkStrategy.RECURSIVE,
        "semantic": ChunkStrategy.SEMANTIC,
        "markdown": ChunkStrategy.MARKDOWN,
        "fixed": ChunkStrategy.FIXED,
        "faq": ChunkStrategy.FAQ,
    }
    return mapping.get(key, ChunkStrategy.RECURSIVE)
