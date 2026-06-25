from typing import Protocol, Optional, List, Dict, Any
from enum import Enum


class DocumentType(str, Enum):
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"
    WORD = "word"
    TEXT = "text"
    JSON = "json"
    CODE = "code"
    LEGAL = "legal"
    TECHNICAL = "technical"


class CleaningLevel(str, Enum):
    LIGHT = "light"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"


class CleanerPort(Protocol):
    def clean(
        self,
        text: str,
        doc_type: DocumentType = DocumentType.TEXT,
        level: CleaningLevel = CleaningLevel.STANDARD,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        ...
    
    def clean_batch(
        self,
        texts: List[str],
        doc_type: DocumentType = DocumentType.TEXT,
        level: CleaningLevel = CleaningLevel.STANDARD
    ) -> List[str]:
        ...
