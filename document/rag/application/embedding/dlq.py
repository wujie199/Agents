"""向量写入失败死信队列（JSONL）。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_log = logging.getLogger("rag.embedding.dlq")


def append_embedding_dlq(
    path: str | Path,
    *,
    collection: str,
    doc_id: str,
    tenant_id: str,
    expected: int,
    error: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "collection": collection,
        "doc_id": doc_id,
        "tenant_id": tenant_id,
        "expected_vectors": expected,
        "error": error,
        **(extra or {}),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    _log.error("向量写入 DLQ: %s doc_id=%s error=%s", target, doc_id, error)
