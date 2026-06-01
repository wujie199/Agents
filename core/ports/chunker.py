from typing import Protocol, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"
    FIXED = "fixed"
    FAQ = "faq"


@dataclass
class Chunk:
    chunk_id: str
    content: str
    doc_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    char_count: int = 0
    token_count: int = 0
    
    def __post_init__(self):
        if self.char_count == 0:
            self.char_count = len(self.content)


class ChunkerPort(Protocol):
    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        ...
    
    def chunk_batch(
        self,
        texts: List[str],
        doc_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[List[Chunk]]:
        ...
