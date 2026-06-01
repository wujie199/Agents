import re
import hashlib
from typing import Optional, List
from enum import Enum
import yaml
from pathlib import Path


class SensitivityLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PII = "pii"
    SECRET = "secret"


class PrivacyPortAdapter:
    def __init__(self, rules_path: Optional[str] = None):
        self._rules = self._load_rules(rules_path)
        self._patterns = self._compile_patterns()
    
    def _load_rules(self, rules_path: Optional[str]) -> dict:
        if rules_path is None:
            return {
                "phone": r"1[3-9]\d{9}",
                "email": r"[\w\.-]+@[\w\.-]+\.\w+",
                "id_card": r"\d{17}[\dXx]",
                "bank_card": r"\d{16,19}",
            }
        
        path = Path(rules_path)
        if not path.exists():
            return {}
        
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    def _compile_patterns(self) -> dict:
        patterns = {}
        for name, pattern in self._rules.items():
            if isinstance(pattern, str):
                patterns[name] = re.compile(pattern)
        return patterns
    
    def mask_text(self, text: str, policy: Optional[str] = None) -> str:
        if not text:
            return text
        
        result = text
        
        if "phone" in self._patterns:
            result = self._patterns["phone"].sub(
                lambda m: m.group()[:3] + "****" + m.group()[-4:],
                result
            )
        
        if "email" in self._patterns:
            result = self._patterns["email"].sub(
                lambda m: m.group()[0] + "***@" + m.group().split("@")[1],
                result
            )
        
        if "id_card" in self._patterns:
            result = self._patterns["id_card"].sub(
                lambda m: m.group()[:6] + "********" + m.group()[-4:],
                result
            )
        
        if "bank_card" in self._patterns:
            result = self._patterns["bank_card"].sub(
                lambda m: m.group()[:4] + "****" + m.group()[-4:],
                result
            )
        
        return result
    
    def redact_for_storage(self, record: dict) -> dict:
        result = {}
        for key, value in record.items():
            if isinstance(value, str):
                result[key] = self.mask_text(value)
            elif isinstance(value, dict):
                result[key] = self.redact_for_storage(value)
            elif isinstance(value, list):
                result[key] = [
                    self.mask_text(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result
    
    def redact_for_llm(
        self,
        messages: List[dict],
        policy: Optional[str] = None
    ) -> List[dict]:
        result = []
        for msg in messages:
            redacted_msg = {"role": msg.get("role", "user")}
            content = msg.get("content", "")
            if isinstance(content, str):
                redacted_msg["content"] = self.mask_text(content)
            elif isinstance(content, list):
                redacted_msg["content"] = [
                    {"type": item.get("type"), "text": self.mask_text(item.get("text", ""))}
                    if item.get("type") == "text" else item
                    for item in content
                ]
            else:
                redacted_msg["content"] = content
            result.append(redacted_msg)
        return result
    
    def hash_for_audit(self, value: str) -> str:
        if not value:
            return ""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    
    def classify_sensitivity(self, text: str) -> SensitivityLevel:
        if not text:
            return SensitivityLevel.PUBLIC
        
        for pattern_name, pattern in self._patterns.items():
            if pattern.search(text):
                if pattern_name in ["id_card", "bank_card"]:
                    return SensitivityLevel.SECRET
                return SensitivityLevel.PII
        
        return SensitivityLevel.PUBLIC
    
    def should_cache(self, text: str) -> bool:
        level = self.classify_sensitivity(text)
        return level != SensitivityLevel.SECRET
    
    def get_cache_ttl(self, text: str, default_ttl: int = 900) -> int:
        level = self.classify_sensitivity(text)
        if level == SensitivityLevel.SECRET:
            return 0
        elif level == SensitivityLevel.PII:
            return default_ttl // 3
        return default_ttl
