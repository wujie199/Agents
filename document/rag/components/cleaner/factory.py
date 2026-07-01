from typing import Dict, Any, Optional

from core.ports.cleaner import DocumentType, CleaningLevel
from document.rag.components.cleaner.composite import CompositeCleaner, CleanerAdapter
from document.rag.components.cleaner.base import (
    WhitespaceCleaner,
    SpecialCharCleaner,
    HtmlCleaner,
    MarkdownCleaner,
    PrivacyCleaner,
    DuplicateCleaner,
    NoiseCleaner,
    EncodingCleaner,
    UnicodeNormalizerCleaner,
    LengthFilterCleaner
)
from document.rag.components.cleaner.domain import (
    LegalDocumentCleaner,
    TechnicalDocCleaner,
    MedicalDocCleaner,
    FinancialDocCleaner,
    AcademicDocCleaner,
    NewsArticleCleaner
)


def build_html_cleaner() -> CompositeCleaner:
    """Module docstring."""
    return CompositeCleaner([
        HtmlCleaner(preserve_links=True),
        SpecialCharCleaner(preserve_chinese=True),
        WhitespaceCleaner(),
        NoiseCleaner(),
        DuplicateCleaner(),
        LengthFilterCleaner(min_length=5, min_meaningful_chars=3),
    ], name="html")


def build_markdown_cleaner() -> CompositeCleaner:
    """Module docstring."""
    return CompositeCleaner([
        MarkdownCleaner(),
        SpecialCharCleaner(preserve_chinese=True),
        WhitespaceCleaner(),
        PrivacyCleaner(mask_ip=False),
        DuplicateCleaner(),
        LengthFilterCleaner(min_length=5),
    ], name="markdown")


def build_legal_cleaner() -> CompositeCleaner:
    """Module docstring."""
    return CompositeCleaner([
        HtmlCleaner(),
        SpecialCharCleaner(preserve_chinese=True, preserve_punctuation=True),
        LegalDocumentCleaner(),
        WhitespaceCleaner(),
        DuplicateCleaner(),
        LengthFilterCleaner(min_length=5, min_meaningful_chars=3),
    ], name="legal")


def build_technical_cleaner() -> CompositeCleaner:
    """Module docstring."""
    return CompositeCleaner([
        MarkdownCleaner(),
        TechnicalDocCleaner(),
        SpecialCharCleaner(preserve_chinese=True),
        WhitespaceCleaner(),
        DuplicateCleaner(),
        LengthFilterCleaner(min_length=5),
    ], name="technical")


def build_medical_cleaner() -> CompositeCleaner:
    """Module docstring."""
    return CompositeCleaner([
        HtmlCleaner(),
        MedicalDocCleaner(),
        PrivacyCleaner(mask_phone=True, mask_email=True, mask_id_card=True),
        SpecialCharCleaner(preserve_chinese=True),
        WhitespaceCleaner(),
        DuplicateCleaner(),
        NoiseCleaner(),
        LengthFilterCleaner(min_length=10),
    ], name="medical")


def build_financial_cleaner() -> CompositeCleaner:
    """Module docstring."""
    return CompositeCleaner([
        HtmlCleaner(),
        FinancialDocCleaner(),
        PrivacyCleaner(mask_phone=True, mask_email=True, mask_id_card=True),
        SpecialCharCleaner(preserve_chinese=True),
        WhitespaceCleaner(),
        DuplicateCleaner(),
        NoiseCleaner(),
        LengthFilterCleaner(min_length=10),
    ], name="financial")


def build_academic_cleaner() -> CompositeCleaner:
    """Module docstring."""
    return CompositeCleaner([
        HtmlCleaner(),
        AcademicDocCleaner(),
        SpecialCharCleaner(preserve_chinese=True),
        WhitespaceCleaner(),
        DuplicateCleaner(),
        LengthFilterCleaner(min_length=20),
    ], name="academic")


def build_news_cleaner() -> CompositeCleaner:
    """Module docstring."""
    return CompositeCleaner([
        HtmlCleaner(),
        NewsArticleCleaner(),
        NoiseCleaner(),
        SpecialCharCleaner(preserve_chinese=True),
        WhitespaceCleaner(),
        DuplicateCleaner(),
        LengthFilterCleaner(min_length=10),
    ], name="news")


def build_default_cleaner() -> CompositeCleaner:
    """Module docstring."""
    return CompositeCleaner([
        UnicodeNormalizerCleaner(),
        HtmlCleaner(preserve_links=False),
        SpecialCharCleaner(preserve_chinese=True, preserve_punctuation=True),
        WhitespaceCleaner(),
        PrivacyCleaner(mask_ip=False),
        NoiseCleaner(),
        DuplicateCleaner(),
        LengthFilterCleaner(min_length=5, min_meaningful_chars=3),
    ], name="default")


def build_enterprise_cleaner() -> CleanerAdapter:
    """Module docstring."""
    
    domain_cleaners = {
        DocumentType.HTML: build_html_cleaner(),
        DocumentType.MARKDOWN: build_markdown_cleaner(),
        DocumentType.LEGAL: build_legal_cleaner(),
        DocumentType.TECHNICAL: build_technical_cleaner(),
        DocumentType.PDF: build_default_cleaner(),
        DocumentType.WORD: build_default_cleaner(),
        DocumentType.TEXT: build_default_cleaner(),
        DocumentType.JSON: build_default_cleaner(),
        DocumentType.CODE: build_technical_cleaner(),
    }
    
    return CleanerAdapter(
        default_cleaner=build_default_cleaner(),
        domain_cleaners=domain_cleaners
    )


def build_cleaner_from_rag_config(cfg: Optional[Any] = None) -> CleanerAdapter:
    """Build cleaner chain from RagPipelineConfig.cleaners YAML block."""
    cleaners_raw = getattr(cfg, "cleaners", None) if cfg is not None else None
    if not cleaners_raw:
        return build_enterprise_cleaner()

    adapter = CleanerAdapter()
    if "default" in cleaners_raw:
        adapter._default_cleaner = adapter.build_cleaner_from_config(cleaners_raw["default"])

    for doc_type_str, cleaner_config in cleaners_raw.get("domains", {}).items():
        try:
            doc_type = DocumentType(doc_type_str)
            cleaner = adapter.build_cleaner_from_config(cleaner_config)
            adapter.register_domain_cleaner(doc_type, cleaner)
        except ValueError:
            continue

    return adapter


def build_cleaner_from_file(config_path: str) -> CleanerAdapter:
    """Module docstring."""
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    adapter = CleanerAdapter()
    
    if "default" in config:
        adapter._default_cleaner = adapter.build_cleaner_from_config(config["default"])
    
    for doc_type_str, cleaner_config in config.get("domains", {}).items():
        try:
            doc_type = DocumentType(doc_type_str)
            cleaner = adapter.build_cleaner_from_config(cleaner_config)
            adapter.register_domain_cleaner(doc_type, cleaner)
        except ValueError:
            continue
    
    return adapter
