# -*- coding: utf-8 -*-
"""PrivacyMiddleware：state 输出脱敏 mask。"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

_logger = logging.getLogger("app.agents.middleware.privacy")

# PII 模式
_PHONE_RE = re.compile(r"(1[3-9]\d{9})|(\+\d{1,3}[-\s]?\d{7,15})")
_ID_CARD_RE = re.compile(r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_BANK_CARD_RE = re.compile(r"[1-9]\d{14,18}")


class PrivacyMiddleware:
    """对 state 中的文本字段做 PII mask。"""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled

    @property
    def name(self) -> str:
        return "privacy"

    async def on_enter(
        self,
        node_name: str,
        state: Any,
        config: Any,
    ) -> Dict[str, Any]:
        return {}

    async def on_exit(
        self,
        node_name: str,
        state: Any,
        config: Any,
        result: Any,
        *,
        error: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._enabled or error:
            return
        # result 中的 assistant_text 做 PII 检测日志（不修改内容，仅审计）
        if isinstance(result, dict):
            text = result.get("assistant_text") or ""
            if text:
                pii_found = self._detect_pii(text)
                if pii_found:
                    _logger.info(
                        "node=%s pii_types=%s action=detected",
                        node_name,
                        list(pii_found),
                    )

    @staticmethod
    def _detect_pii(text: str) -> Dict[str, int]:
        """检测文本中的 PII 类型及数量。"""
        found: Dict[str, int] = {}
        if _PHONE_RE.search(text):
            found["phone"] = len(_PHONE_RE.findall(text))
        if _ID_CARD_RE.search(text):
            found["id_card"] = len(_ID_CARD_RE.findall(text))
        if _EMAIL_RE.search(text):
            found["email"] = len(_EMAIL_RE.findall(text))
        if _BANK_CARD_RE.search(text):
            found["bank_card"] = len(_BANK_CARD_RE.findall(text))
        return found

    @staticmethod
    def mask_pii(text: str) -> str:
        """对文本中的 PII 做 mask 替换。"""
        text = _PHONE_RE.sub(lambda m: m.group()[:3] + "****" + m.group()[-4:], text)
        text = _ID_CARD_RE.sub(lambda m: m.group()[:6] + "********" + m.group()[-4:], text)
        text = _EMAIL_RE.sub(lambda m: m.group()[:2] + "***@" + m.group().split("@")[-1], text)
        text = _BANK_CARD_RE.sub(lambda m: m.group()[:4] + "**********" + m.group()[-4:], text)
        return text
