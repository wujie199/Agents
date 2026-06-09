"""L2 合规：内容哈希与审计日志。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, List, Optional


def content_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


async def append_audit_log(
    db: Any,
    *,
    tenant_id: str,
    user_id: str,
    resource_type: str,
    resource_id: str,
    content_hash: str,
    action: str,
    meta: Optional[dict] = None,
) -> None:
    inserter = getattr(db, "insert_compliance_audit_log", None)
    if inserter is None:
        return
    await inserter(
        {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "content_hash": content_hash,
            "action": action,
            "ts": datetime.now().isoformat(),
            "meta_json": json.dumps(meta or {}, ensure_ascii=False),
        }
    )


async def record_message_audit_before_redact(
    db: Any,
    tenant_id: str,
    user_id: str,
    rows: List[dict],
    *,
    resource_type: str = "message",
    content_key: str = "content",
    id_key: str = "message_id",
) -> List[dict]:
    """为待擦除行生成 content_hash 并写入审计表。"""
    updates: List[dict] = []
    for row in rows:
        raw = row.get(content_key) or ""
        if not raw or raw == "[redacted]":
            continue
        digest = content_sha256(str(raw))
        resource_id = str(row.get(id_key) or "")
        await append_audit_log(
            db,
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            content_hash=digest,
            action="anonymize",
            meta={"session_id": row.get("session_id")},
        )
        updates.append({**row, "content_hash": digest})
    return updates


async def delete_compliance_audit_logs(
    db: Any,
    *,
    tenant_id: str,
    user_id: Optional[str] = None,
    resource_type: Optional[str] = None,
) -> int:
    deleter = getattr(db, "delete_compliance_audit_logs", None)
    if deleter is None:
        return 0
    return await deleter(
        tenant_id=tenant_id,
        user_id=user_id,
        resource_type=resource_type,
    )


async def delete_external_fact_audit_for_user(
    db: Any, tenant_id: str, user_id: str
) -> int:
    return await delete_compliance_audit_logs(
        db,
        tenant_id=tenant_id,
        user_id=user_id,
        resource_type="external_fact",
    )


async def delete_external_fact_audit_for_tenant(db: Any, tenant_id: str) -> int:
    return await delete_compliance_audit_logs(
        db,
        tenant_id=tenant_id,
        resource_type="external_fact",
    )
