# -*- coding: utf-8 -*-
"""审计内容策略：hash / redacted / full。"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal, Optional

from app.agents.middleware.privacy import PrivacyMiddleware

AuditContentMode = Literal["hash", "redacted", "full"]

_VALID_MODES = frozenset({"hash", "redacted", "full"})
_DEFAULT_MAX_CHARS = 8000


def normalize_audit_content_mode(mode: str | None) -> AuditContentMode:
    m = (mode or "hash").strip().lower()
    if m not in _VALID_MODES:
        return "hash"
    return m  # type: ignore[return-value]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n[... truncated ...]", True


def apply_audit_content(
    text: Optional[str],
    mode: AuditContentMode,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """将文本转为审计字段（按策略）。"""
    raw = (text or "").strip()
    if not raw:
        return {"policy": mode, "empty": True}

    if mode == "hash":
        return {"policy": "hash", "sha256": _sha256(raw), "chars": len(raw)}

    body = raw
    if mode == "redacted":
        body = PrivacyMiddleware.mask_pii(raw)
    body, truncated = _truncate(body, max_chars)
    out: dict[str, Any] = {"policy": mode, "text": body, "chars": len(raw)}
    if truncated:
        out["truncated"] = True
    return out


def redact_preview(text: Optional[str], max_len: int = 300) -> str:
    """短预览（用于 memory_summary 等嵌套字段）。"""
    raw = PrivacyMiddleware.mask_pii((text or "").strip())
    if len(raw) <= max_len:
        return raw
    return raw[:max_len] + "..."
