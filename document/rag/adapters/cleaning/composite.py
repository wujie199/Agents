from typing import List, Dict, Any, Optional
import logging

from core.ports.cleaner import CleanerPort, DocumentType, CleaningLevel
from document.rag.adapters.cleaning.base import BaseCleaner


class CompositeCleaner(BaseCleaner):
    """Module docstring."""
    
    def __init__(self, cleaners: List[BaseCleaner], name: str = "composite"):
        super().__init__(name)
        self._cleaners = cleaners
        self._logger.info(f"Composite cleaner initialized with {len(cleaners)} cleaners")
    
    def clean(
        self,
        text: str,
        doc_type: DocumentType = DocumentType.TEXT,
        level: CleaningLevel = CleaningLevel.STANDARD,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        if not text:
            return text
        
        result = text
        for cleaner in self._cleaners:
            try:
                result = cleaner.clean(result, doc_type, level, metadata)
            except Exception as e:
                self._logger.warning(
                    f"Cleaner {cleaner._name} failed: {e}, skipping"
                )
                continue
        
        return result
    
    def add_cleaner(self, cleaner: BaseCleaner) -> None:
        self._cleaners.append(cleaner)
        self._logger.info(f"Added cleaner: {cleaner._name}")
    
    def remove_cleaner(self, name: str) -> bool:
        for i, cleaner in enumerate(self._cleaners):
            if cleaner._name == name:
                self._cleaners.pop(i)
                self._logger.info(f"Removed cleaner: {name}")
                return True
        return False
    
    def get_cleaner_names(self) -> List[str]:
        return [c._name for c in self._cleaners]


class CleanerAdapter:
    """Module docstring."""
    
    def __init__(
        self,
        default_cleaner: Optional[CompositeCleaner] = None,
        domain_cleaners: Optional[Dict[DocumentType, CompositeCleaner]] = None
    ):
        self._default_cleaner = default_cleaner or self._build_default_cleaner()
        self._domain_cleaners = domain_cleaners or {}
        self._logger = logging.getLogger("cleaner.adapter")
    
    def _build_default_cleaner(self) -> CompositeCleaner:
        """Module docstring."""
        from document.rag.adapters.cleaning.base import (
            WhitespaceCleaner,
            SpecialCharCleaner,
            HtmlCleaner,
            PrivacyCleaner,
            DuplicateCleaner,
            NoiseCleaner,
            LengthFilterCleaner
        )
        
        cleaners = [
            HtmlCleaner(preserve_links=False),
            SpecialCharCleaner(preserve_chinese=True, preserve_punctuation=True),
            WhitespaceCleaner(),
            PrivacyCleaner(mask_ip=False),
            NoiseCleaner(),
            DuplicateCleaner(),
            LengthFilterCleaner(min_length=5, min_meaningful_chars=3),
        ]
        
        return CompositeCleaner(cleaners, name="default")
    
    def _get_cleaner_for_type(self, doc_type: DocumentType) -> CompositeCleaner:
        """Module docstring."""
        if doc_type in self._domain_cleaners:
            return self._domain_cleaners[doc_type]
        return self._default_cleaner
    
    def clean(
        self,
        text: str,
        doc_type: DocumentType = DocumentType.TEXT,
        level: CleaningLevel = CleaningLevel.STANDARD,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        cleaner = self._get_cleaner_for_type(doc_type)
        return cleaner.clean(text, doc_type, level, metadata)
    
    def clean_batch(
        self,
        texts: List[str],
        doc_type: DocumentType = DocumentType.TEXT,
        level: CleaningLevel = CleaningLevel.STANDARD
    ) -> List[str]:
        return [
            self.clean(text, doc_type, level)
            for text in texts
        ]
    
    def register_domain_cleaner(
        self,
        doc_type: DocumentType,
        cleaner: CompositeCleaner
    ) -> None:
        self._domain_cleaners[doc_type] = cleaner
        self._logger.info(f"Registered domain cleaner for {doc_type.value}")
    
    def build_cleaner_from_config(self, config: Dict[str, Any]) -> CompositeCleaner:
        """
        config ç¤ºä¾:
        {
            "cleaners": [
                {"type": "whitespace"},
                {"type": "special_char", "preserve_chinese": true},
                {"type": "privacy", "mask_email": true, "mask_phone": true},
                {"type": "html", "preserve_links": false}
            ]
        }
        """
        from document.rag.adapters.cleaning.base import (
            WhitespaceCleaner,
            SpecialCharCleaner,
            HtmlCleaner,
            MarkdownCleaner,
            PrivacyCleaner,
            DuplicateCleaner,
            NoiseCleaner,
            EncodingCleaner,
            LengthFilterCleaner
        )
        from document.rag.adapters.cleaning.domain import (
            LegalDocumentCleaner,
            TechnicalDocCleaner,
            MedicalDocCleaner,
            FinancialDocCleaner,
            AcademicDocCleaner,
            NewsArticleCleaner
        )
        
        cleaner_map = {
            "whitespace": WhitespaceCleaner,
            "special_char": SpecialCharCleaner,
            "html": HtmlCleaner,
            "markdown": MarkdownCleaner,
            "privacy": PrivacyCleaner,
            "duplicate": DuplicateCleaner,
            "noise": NoiseCleaner,
            "encoding": EncodingCleaner,
            "length_filter": LengthFilterCleaner,
            "legal": LegalDocumentCleaner,
            "technical": TechnicalDocCleaner,
            "medical": MedicalDocCleaner,
            "financial": FinancialDocCleaner,
            "academic": AcademicDocCleaner,
            "news": NewsArticleCleaner,
        }
        
        cleaners = []
        for cleaner_config in config.get("cleaners", []):
            cleaner_type = cleaner_config.get("type")
            if cleaner_type not in cleaner_map:
                self._logger.warning(f"Unknown cleaner type: {cleaner_type}")
                continue
            
            cleaner_cls = cleaner_map[cleaner_type]
            params = {k: v for k, v in cleaner_config.items() if k != "type"}
            
            try:
                cleaner = cleaner_cls(**params)
                cleaners.append(cleaner)
            except Exception as e:
                self._logger.warning(
                    f"Failed to create cleaner {cleaner_type}: {e}"
                )
        
        name = config.get("name", "config_driven")
        return CompositeCleaner(cleaners, name=name)
