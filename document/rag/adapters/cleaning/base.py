import re
from typing import Optional, List, Dict, Any
import logging

from core.ports.cleaner import CleanerPort, DocumentType, CleaningLevel


class BaseCleaner:
    def __init__(self, name: str):
        self._name = name
        self._logger = logging.getLogger(f"cleaner.{name}")
    
    def clean(
        self,
        text: str,
        doc_type: DocumentType = DocumentType.TEXT,
        level: CleaningLevel = CleaningLevel.STANDARD,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        raise NotImplementedError
    
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


class WhitespaceCleaner(BaseCleaner):
    """Whitespace normalizer."""
    
    def __init__(self):
        super().__init__("whitespace")
    
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
        
        result = re.sub(r'[ \t]+', ' ', result)
        
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        result = re.sub(r' +', ' ', result)
        
        lines = result.split('\n')
        lines = [line.strip() for line in lines]
        result = '\n'.join(lines)
        
        return result.strip()


class SpecialCharCleaner(BaseCleaner):
    """Remove control/special characters."""
    
    def __init__(
        self,
        preserve_chinese: bool = True,
        preserve_punctuation: bool = True
    ):
        super().__init__("special_char")
        self._preserve_chinese = preserve_chinese
        self._preserve_punctuation = preserve_punctuation
    
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
        
        result = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', result)
        
        if level == CleaningLevel.AGGRESSIVE:
            result = re.sub(r'[\u2000-\u200f\u2028-\u202f\u205f-\u206f]', ' ', result)
        
        if level == CleaningLevel.AGGRESSIVE:
            result = re.sub(r'[\ufeff]', '', result)
        
        if self._preserve_punctuation:
            allowed = r'[^\w\s\u4e00-\u9fffãï¼ï¼ï¼ï¼ï¼ã"''ï¼ï¼ããããââ¦Â·\n\-]'
        else:
            allowed = r'[^\w\s\u4e00-\u9fff\n]'
        
        if level != CleaningLevel.LIGHT:
            result = re.sub(allowed, '', result)
        
        return result


class HtmlCleaner(BaseCleaner):
    """Strip HTML tags."""
    
    def __init__(self, preserve_links: bool = False):
        super().__init__("html")
        self._preserve_links = preserve_links
    
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
        
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(result, "html.parser")
            
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            if self._preserve_links:
                for a in soup.find_all("a"):
                    a.replace_with(f"{a.get_text()} ({a.get('href', '')})")
            
            result = soup.get_text(separator=" ")
            
        except ImportError:
            result = re.sub(r'<script[^>]*>.*</script>', '', result, flags=re.DOTALL | re.IGNORECASE)
            result = re.sub(r'<style[^>]*>.*</style>', '', result, flags=re.DOTALL | re.IGNORECASE)
            result = re.sub(r'<[^>]+>', ' ', result)
        
        result = re.sub(r'&nbsp;', ' ', result)
        result = re.sub(r'&[a-z]+;', '', result)
        result = re.sub(r'&#\d+;', '', result)
        
        return result


class MarkdownCleaner(BaseCleaner):
    """Normalize markdown formatting."""
    
    def __init__(self):
        super().__init__("markdown")
    
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
        
        result = re.sub(r'^#{1,6}\s+', '', result, flags=re.MULTILINE)
        
        result = re.sub(r'\*{1,2}([^\*]+)\*{1,2}', r'\1', result)
        result = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', result)
        
        result = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[å¾ç: \1]', result)
        
        result = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', result)
        
        result = re.sub(r'^[-*+]\s+', '', result, flags=re.MULTILINE)
        result = re.sub(r'^\d+\.\s+', '', result, flags=re.MULTILINE)
        
        result = re.sub(r'`{1,3}([^`]+")`{1,3}', r'\1', result)
        
        result = re.sub(r'^>{1,}\s*', '', result, flags=re.MULTILINE)
        
        result = re.sub(r'^[-]{3,}$', '', result, flags=re.MULTILINE)
        result = re.sub(r'^[*]{3,}$', '', result, flags=re.MULTILINE)
        
        return result


class PrivacyCleaner(BaseCleaner):
    """Mask PII patterns."""
    
    def __init__(
        self,
        mask_phone: bool = True,
        mask_email: bool = True,
        mask_id_card: bool = True,
        mask_bank_card: bool = True,
        mask_ip: bool = False
    ):
        super().__init__("privacy")
        self._mask_phone = mask_phone
        self._mask_email = mask_email
        self._mask_id_card = mask_id_card
        self._mask_bank_card = mask_bank_card
        self._mask_ip = mask_ip
    
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
        
        if self._mask_phone:
            result = re.sub(
                r'1[3-9]\d{9}',
                '[PHONE]',
                result
            )
            result = re.sub(
                r'(\d{3,4})-(\d{7,8})',
                '[PHONE]',
                result
            )
        
        if self._mask_email:
            result = re.sub(
                r'[\w\.-]+@[\w\.-]+\.\w+',
                '[EMAIL]',
                result
            )
        
        if self._mask_id_card:
            result = re.sub(
                r'\d{17}[\dXx]',
                '[ID_CARD]',
                result
            )
            result = re.sub(
                r'\d{15}',
                '[ID_CARD]',
                result
            )
        
        if self._mask_bank_card:
            result = re.sub(
                r'\d{16,19}',
                '[BANK_CARD]',
                result
            )
        
        if self._mask_ip:
            result = re.sub(
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
                '[IP]',
                result
            )
        
        return result


class DuplicateCleaner(BaseCleaner):
    """Deduplicate lines."""
    
    def __init__(
        self,
        min_line_length: int = 10,
        similarity_threshold: float = 0.9
    ):
        super().__init__("duplicate")
        self._min_line_length = min_line_length
        self._similarity_threshold = similarity_threshold
    
    def clean(
        self,
        text: str,
        doc_type: DocumentType = DocumentType.TEXT,
        level: CleaningLevel = CleaningLevel.STANDARD,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        if not text:
            return text
        
        lines = text.split('\n')
        
        seen = set()
        unique_lines = []
        
        for line in lines:
            if len(line.strip()) < self._min_line_length:
                unique_lines.append(line)
                continue
            
            normalized = ' '.join(line.split()).lower()
            
            if normalized not in seen:
                seen.add(normalized)
                unique_lines.append(line)
        
        result = '\n'.join(unique_lines)
        
        result = re.sub(r'(.{20,}")\1+', r'\1', result)
        
        return result


class NoiseCleaner(BaseCleaner):
    """Remove boilerplate noise."""
    
    def __init__(self):
        super().__init__("noise")
        
        self._noise_patterns = [
            (r'Copyright[^\n]*', ''),
            (r'All Rights Reserved\.?', ''),
            (r'\[advertisement\]', '', re.IGNORECASE),
        ]
    
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
        
        for pattern_data in self._noise_patterns:
            if len(pattern_data) == 2:
                pattern, replacement = pattern_data
                flags = 0
            else:
                pattern, replacement, flags = pattern_data
            
            result = re.sub(pattern, replacement, result, flags=flags)
        
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()


class EncodingCleaner(BaseCleaner):
    """Fix common mojibake and fullwidth punctuation."""

    def __init__(self):
        super().__init__("encoding")

    def clean(
        self,
        text: str,
        doc_type: DocumentType = DocumentType.TEXT,
        level: CleaningLevel = CleaningLevel.STANDARD,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not text:
            return text
        result = text
        for src, dst in {
            "，": ",", "。": ".", "：": ":", "；": ";",
            "！": "!", "（": "(", "）": ")",
            "“": '"', "”": '"', "‘": "'", "’": "'",
        }.items():
            result = result.replace(src, dst)
        result = re.sub(r'\\u[0-9a-fA-F]{4}', '', result)
        result = re.sub(r'\\x[0-9a-fA-F]{2}', '', result)
        return result

class LengthFilterCleaner(BaseCleaner):
    """Filter by length."""
    
    def __init__(
        self,
        min_length: int = 10,
        max_length: Optional[int] = None,
        min_meaningful_chars: int = 5
    ):
        super().__init__("length_filter")
        self._min_length = min_length
        self._max_length = max_length
        self._min_meaningful_chars = min_meaningful_chars
    
    def clean(
        self,
        text: str,
        doc_type: DocumentType = DocumentType.TEXT,
        level: CleaningLevel = CleaningLevel.STANDARD,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        if not text:
            return text
        
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            if len(stripped) < self._min_length:
                continue
            
            meaningful_chars = len(re.sub(r'[\s\-\.\,\ï¼\ã]+', '', stripped))
            if meaningful_chars < self._min_meaningful_chars:
                continue
            
            filtered_lines.append(line)
        
        result = '\n'.join(filtered_lines)
        
        if self._max_length and len(result) > self._max_length:
            result = result[:self._max_length]
            last_newline = result.rfind('\n')
            if last_newline > self._max_length * 0.8:
                result = result[:last_newline]
            result += '\n[truncated]'
        
        return result
