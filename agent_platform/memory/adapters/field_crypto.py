"""可选字段级加密（冷归档对象 / 敏感 payload）。"""

from __future__ import annotations

import base64
import hashlib
import os
from typing import Any, Optional


def _derive_fernet_key(raw: str) -> bytes:
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_bytes(data: bytes, key: str) -> bytes:
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise RuntimeError(
            "cold_archive_encrypt_at_rest requires: pip install cryptography"
        ) from exc

    token = Fernet(_derive_fernet_key(key)).encrypt(data)
    return b"ENC1:" + token


def decrypt_bytes(data: bytes, key: str) -> bytes:
    if not data.startswith(b"ENC1:"):
        return data
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise RuntimeError(
            "cold_archive_encrypt_at_rest requires: pip install cryptography"
        ) from exc

    token = data[5:]
    return Fernet(_derive_fernet_key(key)).decrypt(token)


def resolve_encryption_key(
    explicit: Optional[str] = None,
    secret_port: Any = None,
) -> Optional[str]:
    if explicit:
        return explicit
    env_key = os.environ.get("MEMORY_ENCRYPTION_KEY")
    if env_key:
        return env_key
    if secret_port is not None:
        getter = getattr(secret_port, "get_secret", None)
        if getter is not None:
            return getter("memory_encryption_key")
    return None
