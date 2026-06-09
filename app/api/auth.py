# -*- coding: utf-8 -*-
"""HTTP API 鉴权（可选 API Key）。"""

from __future__ import annotations

import os
from typing import Optional

try:
    from fastapi import Header, HTTPException, Request
except ImportError:  # pragma: no cover
    Header = None  # type: ignore[misc, assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    Request = object  # type: ignore[misc, assignment]


def configured_api_key() -> Optional[str]:
    return os.environ.get("CHAT_API_KEY") or None


def configured_memory_admin_key() -> Optional[str]:
    return (
        os.environ.get("MEMORY_ADMIN_API_KEY")
        or os.environ.get("CHAT_API_KEY")
        or None
    )


def verify_api_key(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    expected = configured_api_key()
    if not expected:
        return
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="无效 API Key")


def verify_memory_admin_key(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """记忆管理敏感操作（purge/retention）鉴权。"""
    expected = configured_memory_admin_key()
    if not expected:
        return
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="无效 Memory Admin API Key")
