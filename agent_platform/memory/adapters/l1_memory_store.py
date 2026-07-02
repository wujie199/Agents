# -*- coding: utf-8 -*-
"""Hermes 风格 L1 记忆：§ 分隔条目、双态（冻结快照 + 实时磁盘）、漂移检测。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.domain.context import RequestContext
from core.ports.memory import PromptMemorySnapshot

from agent_platform.memory.adapters.file_lock import file_lock
from agent_platform.memory.adapters.hot_memory_file_adapter import HotMemoryFileAdapter
from agent_platform.memory.adapters.memory_security import scan_memory_content

logger = logging.getLogger(__name__)

ENTRY_DELIMITER = "\n§\n"
OnMemoryWriteCallback = Callable[[str, str, str, dict[str, Any]], None]


def _drift_error(path: Path, bak_path: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": (
            f"Refusing to write {path.name}: file on disk has content that "
            f"wouldn't round-trip through the memory tool (external edit or "
            f"concurrent session). Snapshot saved to {bak_path}. "
            f"Rewrite the file as a clean §-delimited entry list, then retry."
        ),
        "drift_backup": bak_path,
        "remediation": (
            "Integrate missing content via memory(action=add), then clean the file."
        ),
    }


class L1MemoryStore:
    """§ 分隔的 MEMORY.md / USER_{user_id}.md，会话级冻结快照由 MemoryPortAdapter 缓存。"""

    def __init__(
        self,
        store_dir: str = "workspace/memory",
        memory_char_limit: int = 2200,
        user_char_limit: int = 1375,
        *,
        use_file_lock: bool = True,
        on_memory_write: Optional[OnMemoryWriteCallback] = None,
        hot_adapter: Optional[HotMemoryFileAdapter] = None,
    ):
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self._on_memory_write = on_memory_write
        self._hot = hot_adapter or HotMemoryFileAdapter(
            store_dir=store_dir,
            hot_memory_max_chars=memory_char_limit,
            user_memory_max_chars=user_char_limit,
            use_file_lock=use_file_lock,
        )
        self._use_file_lock = self._hot._use_file_lock
        self._live: dict[str, dict[str, list[str]]] = {}

    def _scope_key(self, tenant_id: str, user_id: str) -> str:
        return f"{tenant_id}:{user_id}"

    def _memory_path(self, tenant_id: str) -> Path:
        return self._hot._memory_path(tenant_id)

    def _user_path(self, tenant_id: str, user_id: str) -> Path:
        return self._hot._user_path(tenant_id, user_id)

    def _path_for(self, tenant_id: str, user_id: str, target: str) -> Path:
        if target == "user":
            return self._user_path(tenant_id, user_id)
        return self._memory_path(tenant_id)

    def _char_limit(self, target: str) -> int:
        return self.user_char_limit if target == "user" else self.memory_char_limit

    def _get_live(self, tenant_id: str, user_id: str) -> dict[str, list[str]]:
        key = self._scope_key(tenant_id, user_id)
        if key not in self._live:
            self.load_from_disk(tenant_id, user_id)
        return self._live[key]

    def _entries_for(self, tenant_id: str, user_id: str, target: str) -> list[str]:
        return list(self._get_live(tenant_id, user_id)[target])

    def _set_entries(
        self, tenant_id: str, user_id: str, target: str, entries: list[str]
    ) -> None:
        self._get_live(tenant_id, user_id)[target] = entries

    @staticmethod
    def _read_file(path: Path) -> list[str]:
        if not path.is_file():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return []
        if not raw.strip():
            return []
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    @staticmethod
    def _write_file(path: Path, entries: list[str], *, use_file_lock: bool) -> None:
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        HotMemoryFileAdapter._atomic_write(path, content, use_file_lock=use_file_lock)

    def _char_count(self, tenant_id: str, user_id: str, target: str) -> int:
        entries = self._entries_for(tenant_id, user_id, target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    @staticmethod
    def _sanitize_entries_for_snapshot(
        entries: list[str], filename: str
    ) -> list[str]:
        sanitized: list[str] = []
        for entry in entries:
            if not entry or entry.startswith("[BLOCKED:"):
                sanitized.append(entry)
                continue
            threat = scan_memory_content(entry)
            if threat:
                logger.warning(
                    "Memory entry from %s blocked at load: %s", filename, threat
                )
                sanitized.append(
                    f"[BLOCKED: {filename} entry contained threat pattern: "
                    f"{threat}. Removed from system prompt.]"
                )
            else:
                sanitized.append(entry)
        return sanitized

    @staticmethod
    def _render_block(target: str, entries: list[str], char_limit: int) -> str:
        if not entries:
            return ""
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / char_limit) * 100)) if char_limit > 0 else 0
        if target == "user":
            header = (
                f"USER PROFILE (who the user is) [{pct}% — "
                f"{current:,}/{char_limit:,} chars]"
            )
        else:
            header = (
                f"MEMORY (your personal notes) [{pct}% — "
                f"{current:,}/{char_limit:,} chars]"
            )
        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"

    def load_from_disk(self, tenant_id: str, user_id: str) -> dict[str, list[str]]:
        memory_entries = self._read_file(self._memory_path(tenant_id))
        user_entries = self._read_file(self._user_path(tenant_id, user_id))
        memory_entries = list(dict.fromkeys(memory_entries))
        user_entries = list(dict.fromkeys(user_entries))
        state = {"memory": memory_entries, "user": user_entries}
        self._live[self._scope_key(tenant_id, user_id)] = state
        return state

    def capture_frozen_snapshot(
        self, context: RequestContext
    ) -> PromptMemorySnapshot:
        tenant_id = context.tenant_id
        user_id = context.user_id
        live = self.load_from_disk(tenant_id, user_id)
        sanitized_memory = self._sanitize_entries_for_snapshot(
            live["memory"], "MEMORY.md"
        )
        sanitized_user = self._sanitize_entries_for_snapshot(
            live["user"], f"USER_{user_id}.md"
        )
        memory_block = self._render_block(
            "memory", sanitized_memory, self.memory_char_limit
        )
        user_block = self._render_block(
            "user", sanitized_user, self.user_char_limit
        )
        parts = []
        if memory_block:
            parts.append(f"# SYSTEM MEMORY\n\n{memory_block}")
        else:
            parts.append("# SYSTEM MEMORY\n\n")
        if user_block:
            parts.append(f"# USER PREFERENCES\n\n{user_block}")
        else:
            parts.append("# USER PREFERENCES\n\n")
        combined = "\n\n".join(parts)
        snapshot_hash = HotMemoryFileAdapter._compute_hash(combined)
        return PromptMemorySnapshot(
            memory_text=combined,
            hash=snapshot_hash,
            frozen=True,
        )

    def _detect_external_drift(
        self, tenant_id: str, user_id: str, target: str
    ) -> Optional[str]:
        path = self._path_for(tenant_id, user_id, target)
        if not path.is_file():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        if not raw.strip():
            return None
        parsed = [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]
        roundtrip = ENTRY_DELIMITER.join(parsed)
        char_limit = self._char_limit(target)
        max_entry_len = max((len(e) for e in parsed), default=0)
        drift = (raw.strip() != roundtrip) or (max_entry_len > char_limit)
        if not drift:
            return None
        ts = int(time.time())
        bak_path = path.with_suffix(path.suffix + f".bak.{ts}")
        try:
            bak_path.write_text(raw, encoding="utf-8")
        except OSError:
            return str(bak_path) + " (BACKUP FAILED)"
        return str(bak_path)

    def _reload_target(
        self, tenant_id: str, user_id: str, target: str
    ) -> Optional[str]:
        bak = self._detect_external_drift(tenant_id, user_id, target)
        path = self._path_for(tenant_id, user_id, target)
        fresh = self._read_file(path)
        fresh = list(dict.fromkeys(fresh))
        self._set_entries(tenant_id, user_id, target, fresh)
        return bak

    def _save_to_disk(self, tenant_id: str, user_id: str, target: str) -> None:
        path = self._path_for(tenant_id, user_id, target)
        entries = self._entries_for(tenant_id, user_id, target)
        self._write_file(path, entries, use_file_lock=self._use_file_lock)
        if target == "memory":
            self._hot._memory_cache[tenant_id] = (
                ENTRY_DELIMITER.join(entries) if entries else ""
            )
        else:
            cache_key = f"{tenant_id}:{user_id}"
            self._hot._user_cache[cache_key] = (
                ENTRY_DELIMITER.join(entries) if entries else ""
            )

    def _notify_write(
        self, tenant_id: str, user_id: str, action: str, payload: dict[str, Any]
    ) -> None:
        if self._on_memory_write is None:
            return
        try:
            self._on_memory_write(tenant_id, user_id, action, payload)
        except Exception as exc:
            logger.warning("on_memory_write callback failed: %s", exc)

    def _success_response(
        self,
        tenant_id: str,
        user_id: str,
        target: str,
        message: str | None = None,
        *,
        done: bool = False,
    ) -> dict[str, Any]:
        entries = self._entries_for(tenant_id, user_id, target)
        current = self._char_count(tenant_id, user_id, target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        resp: dict[str, Any] = {
            "success": True,
            "target": target,
            "entries": entries,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        if done:
            resp["done"] = True
        return resp

    def _budget_error(
        self,
        tenant_id: str,
        user_id: str,
        target: str,
        *,
        new_total: int,
        extra_msg: str,
    ) -> dict[str, Any]:
        entries = self._entries_for(tenant_id, user_id, target)
        current = self._char_count(tenant_id, user_id, target)
        limit = self._char_limit(target)
        return {
            "success": False,
            "error": (
                f"Memory at {current:,}/{limit:,} chars. {extra_msg} "
                f"Consolidate now: use 'replace' to merge overlapping entries "
                f"into shorter ones or 'remove' stale entries (see current_entries), "
                f"then retry — all in this turn."
            ),
            "current_entries": entries,
            "usage": f"{current:,}/{limit:,}",
            "would_be": f"{new_total:,}/{limit:,}",
        }

    def _with_file_lock(self, path: Path):
        if self._use_file_lock:
            return file_lock(path, exclusive=True)
        from contextlib import nullcontext

        return nullcontext()

    def add(
        self, tenant_id: str, user_id: str, target: str, content: str
    ) -> dict[str, Any]:
        content = (content or "").strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}
        threat = scan_memory_content(content)
        if threat:
            return {"success": False, "error": threat}

        path = self._path_for(tenant_id, user_id, target)
        with self._with_file_lock(path):
            bak = self._reload_target(tenant_id, user_id, target)
            if bak:
                return _drift_error(path, bak)

            entries = self._entries_for(tenant_id, user_id, target)
            if content in entries:
                return self._success_response(
                    tenant_id,
                    user_id,
                    target,
                    "Entry already exists (no duplicate added).",
                )
            new_entries = entries + [content]
            new_total = len(ENTRY_DELIMITER.join(new_entries))
            limit = self._char_limit(target)
            if new_total > limit:
                return self._budget_error(
                    tenant_id,
                    user_id,
                    target,
                    new_total=new_total,
                    extra_msg=f"Adding this entry ({len(content)} chars) would exceed the limit.",
                )
            entries.append(content)
            self._set_entries(tenant_id, user_id, target, entries)
            self._save_to_disk(tenant_id, user_id, target)

        self._notify_write(
            tenant_id, user_id, "add", {"target": target, "content": content}
        )
        return self._success_response(tenant_id, user_id, target, "Entry added.")

    def replace(
        self,
        tenant_id: str,
        user_id: str,
        target: str,
        old_text: str,
        new_content: str,
    ) -> dict[str, Any]:
        old_text = (old_text or "").strip()
        new_content = (new_content or "").strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {
                "success": False,
                "error": "new_content cannot be empty. Use 'remove' to delete entries.",
            }
        threat = scan_memory_content(new_content)
        if threat:
            return {"success": False, "error": threat}

        path = self._path_for(tenant_id, user_id, target)
        with self._with_file_lock(path):
            bak = self._reload_target(tenant_id, user_id, target)
            if bak:
                return _drift_error(path, bak)

            entries = self._entries_for(tenant_id, user_id, target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]
            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}
            if len(matches) > 1:
                unique = {e for _, e in matches}
                if len(unique) > 1:
                    previews = [
                        e[:80] + ("..." if len(e) > 80 else "") for _, e in matches
                    ]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
            idx = matches[0][0]
            test_entries = entries.copy()
            test_entries[idx] = new_content
            new_total = len(ENTRY_DELIMITER.join(test_entries))
            limit = self._char_limit(target)
            if new_total > limit:
                return self._budget_error(
                    tenant_id,
                    user_id,
                    target,
                    new_total=new_total,
                    extra_msg=f"Replacement would put memory at {new_total:,}/{limit:,} chars.",
                )
            entries[idx] = new_content
            self._set_entries(tenant_id, user_id, target, entries)
            self._save_to_disk(tenant_id, user_id, target)

        self._notify_write(
            tenant_id,
            user_id,
            "replace",
            {"target": target, "old_text": old_text, "content": new_content},
        )
        return self._success_response(tenant_id, user_id, target, "Entry replaced.")

    def remove(
        self, tenant_id: str, user_id: str, target: str, old_text: str
    ) -> dict[str, Any]:
        old_text = (old_text or "").strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        path = self._path_for(tenant_id, user_id, target)
        with self._with_file_lock(path):
            bak = self._reload_target(tenant_id, user_id, target)
            if bak:
                return _drift_error(path, bak)

            entries = self._entries_for(tenant_id, user_id, target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]
            if not matches:
                return {"success": False, "error": f"No entry matched '{old_text}'."}
            if len(matches) > 1:
                unique = {e for _, e in matches}
                if len(unique) > 1:
                    previews = [
                        e[:80] + ("..." if len(e) > 80 else "") for _, e in matches
                    ]
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
            entries.pop(matches[0][0])
            self._set_entries(tenant_id, user_id, target, entries)
            self._save_to_disk(tenant_id, user_id, target)

        self._notify_write(
            tenant_id, user_id, "remove", {"target": target, "old_text": old_text}
        )
        return self._success_response(tenant_id, user_id, target, "Entry removed.")

    def apply_batch(
        self,
        tenant_id: str,
        user_id: str,
        target: str,
        operations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not operations:
            return {"success": False, "error": "operations cannot be empty."}

        path = self._path_for(tenant_id, user_id, target)
        with self._with_file_lock(path):
            bak = self._reload_target(tenant_id, user_id, target)
            if bak:
                return _drift_error(path, bak)

            entries = self._entries_for(tenant_id, user_id, target)
            limit = self._char_limit(target)

            for op in operations:
                action = str(op.get("action") or "").strip()
                if action == "add":
                    content = str(op.get("content") or "").strip()
                    if not content:
                        return {"success": False, "error": "add requires content."}
                    threat = scan_memory_content(content)
                    if threat:
                        return {"success": False, "error": threat}
                    if content not in entries:
                        entries.append(content)
                elif action == "replace":
                    old_text = str(op.get("old_text") or "").strip()
                    content = str(op.get("content") or "").strip()
                    if not old_text or not content:
                        return {
                            "success": False,
                            "error": "replace requires old_text and content.",
                        }
                    threat = scan_memory_content(content)
                    if threat:
                        return {"success": False, "error": threat}
                    matches = [(i, e) for i, e in enumerate(entries) if old_text in e]
                    if not matches:
                        return {
                            "success": False,
                            "error": f"No entry matched '{old_text}'.",
                        }
                    if len(matches) > 1:
                        unique = {e for _, e in matches}
                        if len(unique) > 1:
                            return {
                                "success": False,
                                "error": f"Multiple entries matched '{old_text}'.",
                            }
                    entries[matches[0][0]] = content
                elif action == "remove":
                    old_text = str(op.get("old_text") or "").strip()
                    if not old_text:
                        return {"success": False, "error": "remove requires old_text."}
                    matches = [(i, e) for i, e in enumerate(entries) if old_text in e]
                    if not matches:
                        return {
                            "success": False,
                            "error": f"No entry matched '{old_text}'.",
                        }
                    if len(matches) > 1:
                        unique = {e for _, e in matches}
                        if len(unique) > 1:
                            return {
                                "success": False,
                                "error": f"Multiple entries matched '{old_text}'.",
                            }
                    entries.pop(matches[0][0])
                else:
                    return {"success": False, "error": f"Unknown batch action '{action}'."}

            entries = list(dict.fromkeys(entries))
            new_total = len(ENTRY_DELIMITER.join(entries)) if entries else 0
            if new_total > limit:
                return self._budget_error(
                    tenant_id,
                    user_id,
                    target,
                    new_total=new_total,
                    extra_msg="Batch result would exceed the limit.",
                )
            self._set_entries(tenant_id, user_id, target, entries)
            self._save_to_disk(tenant_id, user_id, target)

        self._notify_write(
            tenant_id,
            user_id,
            "batch",
            {"target": target, "operations": operations},
        )
        return self._success_response(
            tenant_id,
            user_id,
            target,
            "Batch applied.",
            done=True,
        )

    def invoke_memory_tool(
        self,
        context: RequestContext,
        *,
        action: str,
        target: str = "memory",
        content: str | None = None,
        old_text: str | None = None,
        operations: list[dict[str, Any]] | None = None,
        write_approval: bool = False,
    ) -> dict[str, Any]:
        tenant_id = context.tenant_id
        user_id = context.user_id
        if target not in ("memory", "user"):
            return {"success": False, "error": f"Invalid target '{target}'."}

        if operations:
            if write_approval:
                self._queue_pending_op(
                    tenant_id, user_id,
                    {"action": "batch", "target": target, "operations": operations},
                )
                return {
                    "success": True,
                    "staged": True,
                    "message": "Batch staged for approval.",
                }
            return self.apply_batch(tenant_id, user_id, target, operations)

        action = (action or "").strip()
        if action == "add" and not content:
            return {"success": False, "error": "Content is required for 'add'."}
        if action == "replace" and (not old_text or not content):
            missing = "old_text" if not old_text else "content"
            return {"success": False, "error": f"{missing} is required for 'replace'."}
        if action == "remove" and not old_text:
            return {"success": False, "error": "old_text is required for 'remove'."}
        if action not in ("add", "replace", "remove"):
            return {
                "success": False,
                "error": f"Unknown action '{action}'. Use: add, replace, remove.",
            }

        if write_approval:
            payload = {
                "action": action,
                "target": target,
                "content": content,
                "old_text": old_text,
            }
            self._queue_pending_op(tenant_id, user_id, payload)
            return {
                "success": True,
                "staged": True,
                "message": f"Memory {action} staged for approval.",
            }

        if action == "add":
            return self.add(tenant_id, user_id, target, content or "")
        if action == "replace":
            return self.replace(
                tenant_id, user_id, target, old_text or "", content or ""
            )
        return self.remove(tenant_id, user_id, target, old_text or "")

    def _queue_pending_op(
        self, tenant_id: str, user_id: str, payload: dict[str, Any]
    ) -> None:
        path = self._hot._pending_path(tenant_id, user_id)
        line = json.dumps({"kind": "memory_tool", **payload}, ensure_ascii=False)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        path.write_text(
            (existing + line + "\n") if existing else (line + "\n"),
            encoding="utf-8",
        )

    def flush_pending_memory_ops(
        self, tenant_id: str, user_id: str
    ) -> list[dict[str, Any]]:
        path = self._hot._pending_path(tenant_id, user_id)
        if not path.is_file():
            return []
        remaining: list[str] = []
        applied: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("kind") != "memory_tool":
                remaining.append(line)
                continue
            if data.get("action") == "batch":
                result = self.apply_batch(
                    tenant_id,
                    user_id,
                    data.get("target", "memory"),
                    data.get("operations") or [],
                )
            else:
                ctx = RequestContext(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    session_id="pending",
                    trace_id="pending",
                    channel="system",
                )
                result = self.invoke_memory_tool(
                    ctx,
                    action=data.get("action", ""),
                    target=data.get("target", "memory"),
                    content=data.get("content"),
                    old_text=data.get("old_text"),
                    write_approval=False,
                )
            applied.append(result)
        if remaining:
            path.write_text("\n".join(remaining) + "\n", encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
        return applied

    def get_live_entries(
        self, tenant_id: str, user_id: str, target: str
    ) -> list[str]:
        return self._entries_for(tenant_id, user_id, target)
