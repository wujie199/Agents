"""L3 Skill 执行审计（L2 archive DB）。"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, List, Optional


async def record_skill_run(
    archive_db: Any,
    *,
    skill_id: str,
    tenant_id: str,
    user_id: str,
    session_id: str,
    trace_id: str,
    success: bool,
    steps_executed: int = 0,
    error: Optional[str] = None,
    inputs: Optional[dict] = None,
    outputs: Optional[dict] = None,
) -> Optional[str]:
    if archive_db is None:
        return None
    inserter = getattr(archive_db, "insert_skill_run", None)
    if inserter is None:
        return None
    return await inserter(
        {
            "run_id": str(uuid.uuid4()),
            "skill_id": skill_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "session_id": session_id,
            "trace_id": trace_id,
            "success": 1 if success else 0,
            "steps_executed": steps_executed,
            "error": error,
            "inputs_json": json.dumps(inputs or {}, ensure_ascii=False, default=str),
            "outputs_json": json.dumps(
                outputs or {}, ensure_ascii=False, default=str
            )[:8000],
            "ts": datetime.now().isoformat(),
        }
    )


async def list_skill_runs(
    archive_db: Any,
    *,
    tenant_id: str,
    user_id: Optional[str] = None,
    skill_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> List[dict]:
    if archive_db is None:
        return []
    lister = getattr(archive_db, "list_skill_runs", None)
    if lister is None:
        return []
    return await lister(
        tenant_id=tenant_id,
        user_id=user_id,
        skill_id=skill_id,
        limit=limit,
        offset=offset,
    )


async def delete_skill_runs_for_user(
    archive_db: Any, tenant_id: str, user_id: str
) -> int:
    if archive_db is None:
        return 0
    deleter = getattr(archive_db, "delete_skill_runs_for_user", None)
    if deleter is None:
        return 0
    return await deleter(tenant_id, user_id)


async def delete_skill_runs_for_tenant(archive_db: Any, tenant_id: str) -> int:
    if archive_db is None:
        return 0
    deleter = getattr(archive_db, "delete_skill_runs_for_tenant", None)
    if deleter is None:
        return 0
    return await deleter(tenant_id)
