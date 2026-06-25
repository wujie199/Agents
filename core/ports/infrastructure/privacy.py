from typing import Protocol, Any, Optional, List
from enum import Enum


class SensitivityLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PII = "pii"
    SECRET = "secret"


class PrivacyPort(Protocol):
    def mask_text(self, text: str, policy: Optional[str] = None) -> str:
        ...

    def redact_for_storage(self, record: dict) -> dict:
        ...

    def redact_for_llm(
        self,
        messages: List[dict],
        policy: Optional[str] = None
    ) -> List[dict]:
        ...

    def hash_for_audit(self, value: str) -> str:
        ...

    def classify_sensitivity(self, text: str) -> SensitivityLevel:
        ...
