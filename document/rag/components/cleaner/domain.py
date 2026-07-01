import re
from typing import Optional, List, Dict, Any
import logging

from core.ports.cleaner import CleanerPort, DocumentType, CleaningLevel
from document.rag.components.cleaner.base import BaseCleaner


class LegalDocumentCleaner(BaseCleaner):
    """Legal document structure cleaner."""

    def __init__(self):
        super().__init__("legal")

        self._article_pattern = re.compile(
            r"第[一二三四五六七八九十百千]+条"
        )
        self._chapter_pattern = re.compile(
            r"第[一二三四五六七八九十百千]+章"
        )
    
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
        
        result = re.sub(r"（\d{4}）\s*", "", result)
        result = re.sub(
            r"(\S)(第[一二三四五六七八九十百千零\d]+条)",
            r"\1\n\2",
            result,
        )
        result = re.sub(
            r"(\S)(第[一二三四五六七八九十百千零\d]+章)",
            r"\1\n\n\2",
            result,
        )
        result = re.sub(
            r"[（(]\s*([一二三四五六七八九十百千]+)\s*[)）]",
            r"第\1条",
            result,
        )
        result = re.sub(r"签订地点[^\n]*", "", result)
        result = re.sub(r"签订日期[^\n]*", "", result)
        
        return result.strip()


class TechnicalDocCleaner(BaseCleaner):
    """Technical document cleaner (code blocks, images, constants)."""
    
    def __init__(self):
        super().__init__("technical")
    
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
        
        result = re.sub(r"```[a-z]*\n", "\n[代码块开始]\n", result)
        result = re.sub(r"```", "\n[代码块结束]\n", result)
        result = re.sub(r"`([^`]+)`", r"代码:\1", result)
        result = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"[图示:\1]", result)
        result = re.sub(
            r"[A-Z_]{3,}",
            lambda m: f"[常量:{m.group()}]",
            result,
        )
        
        result = re.sub(
            r'version:\s*[\d.]+',
            '',
            result,
            flags=re.IGNORECASE
        )
        
        return result.strip()


class MedicalDocCleaner(BaseCleaner):
    """Medical document PII masking."""
    
    def __init__(self):
        super().__init__("medical")
    
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
        
        result = re.sub(r"门诊号[：:]\s*[A-Z0-9]+", "[门诊号]", result)
        result = re.sub(r"病历号[：:]\s*[A-Z0-9]+", "[病历号]", result)
        result = re.sub(r"床号[：:]\s*[A-Z0-9]+", "[床号]", result)
        result = re.sub(
            r"患者[姓名]*[：:]\s*[^\s：:\n]+",
            "患者[姓名已脱敏]",
            result,
        )
        result = re.sub(r"年龄[：:]\s*\d+岁", "[年龄]", result)
        result = re.sub(r"\d{4}[-年]\d{1,2}[-月]\d{1,2}日", "[日期]", result)
        
        return result.strip()


class FinancialDocCleaner(BaseCleaner):
    """Module helper."""
    
    def __init__(self):
        super().__init__("financial")
    
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
        
        result = re.sub(r"账号[：:]\s*[\dA-Z]+", "[账号]", result)
        result = re.sub(r"账户号[：:]\s*[\dA-Z]+", "[账户号]", result)
        result = re.sub(r"交易[流水号]*[：:]\s*[A-Z0-9]+", "[交易号]", result)
        result = re.sub(r"¥\s*[\d,]+\.\d*", "[金额]", result)
        result = re.sub(r"人民币\s*[\d,]+\.\d*\s*元", "[金额]", result)
        result = re.sub(
            r"身份证[：:]\s*\d{17}[\dXx]",
            "[身份证]",
            result,
        )
        
        return result.strip()


class AcademicDocCleaner(BaseCleaner):
    """Module helper."""
    
    def __init__(self):
        super().__init__("academic")
    
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
        
        result = re.sub(r"\[\d+\]", "[引用]", result)
        result = re.sub(r"\([A-Z][^)]+,\s*\d{4}[a-z]\)", "[引用]", result)
        result = re.sub(r"关键词[：:]\s*", "\n关键词 ", result)
        result = re.sub(r"摘要[：:]\s*", "\n摘要: ", result)
        result = re.sub(r"DOI[：:]\s*[^\s]+", "[DOI]", result, flags=re.IGNORECASE)
        result = re.sub(r"通讯作者[：:][^\n]+", "", result)
        
        return result.strip()


class NewsArticleCleaner(BaseCleaner):
    """Module helper."""
    
    def __init__(self):
        super().__init__("news")
    
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
        
        result = re.sub(r"记者[：:]\s*[^\s\n]+", "", result)
        result = re.sub(r"编辑[：:]\s*[^\s\n]+", "", result)
        result = re.sub(r"来源[：:]\s*[^\s\n]+", "", result)
        result = re.sub(
            r"\d{4}年\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2}",
            "[发布时间]",
            result,
        )
        result = re.sub(r"（[^）]+）", "", result)
        result = re.sub(r"相关阅读[：:][^\n]*", "", result)
        result = re.sub(r"推荐阅读[：:][^\n]*", "", result)
        
        return result.strip()
