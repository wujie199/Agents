# -*- coding: utf-8 -*-
"""审计事件持久化：append-only NDJSON（fail-open）。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_logger = logging.getLogger("app.agents.middleware.audit_persist")


def persist_audit_event(entry: Dict[str, Any], audit_dir: str | Path) -> None:
    """追加写入 audit_YYYY-MM-DD.jsonl；失败时仅打日志。"""
    try:
        root = Path(audit_dir)
        root.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = root / f"audit_{day}.jsonl"
        payload = dict(entry)
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        _logger.warning("audit persist fail-open: %s", exc)
