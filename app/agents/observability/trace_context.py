# -*- coding: utf-8 -*-
"""HTTP trace_id 解析。"""

from __future__ import annotations

import uuid
from typing import Optional


def resolve_trace_id(
    *,
    header_value: Optional[str] = None,
    traceparent: Optional[str] = None,
    body_trace_id: Optional[str] = None,
    fallback: Optional[str] = None,
) -> str:
    """从 HTTP 头 / traceparent / body 解析 trace_id。"""
    if header_value and header_value.strip():
        return header_value.strip()
    if traceparent:
        parts = traceparent.strip().split("-")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    if body_trace_id and body_trace_id.strip() and body_trace_id.strip() != "api":
        return body_trace_id.strip()
    if fallback and fallback.strip():
        return fallback.strip()
    return str(uuid.uuid4())
