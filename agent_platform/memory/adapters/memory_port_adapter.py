import asyncio
import json
import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import logging

from core.domain.context import RequestContext
from core.ports.memory import (
    MemoryPort,
    PromptMemorySnapshot,
    TurnRecord,
    MemoryDelta,
    SkillSummary
)
from core.ports.storage.relational import RelationalPort


class MemoryPortAdapter:
    def __init__(
        self,
        store_dir: str = "workspace/memory",
        archive_db: Optional[RelationalPort] = None,
        hot_memory_max_chars: int = 2200,
        user_memory_max_chars: int = 1375
    ):
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        
        self._archive_db = archive_db
        self._hot_memory_max = hot_memory_max_chars
        self._user_memory_max = user_memory_max_chars
        self._logger = logging.getLogger(__name__)
        
        self._memory_cache: Dict[str, str] = {}
        self._user_cache: Dict[str, str] = {}
        self._snapshot_hash: Dict[str, str] = {}
    
    def _get_memory_path(self, tenant_id: str) -> Path:
        path = self._store_dir / tenant_id / "MEMORY.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def _get_user_path(self, tenant_id: str, user_id: str) -> Path:
        path = self._store_dir / tenant_id / f"USER_{user_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def _load_memory(self, tenant_id: str) -> str:
        if tenant_id in self._memory_cache:
            return self._memory_cache[tenant_id]
        
        path = self._get_memory_path(tenant_id)
        if path.exists():
            content = path.read_text(encoding="utf-8")
        else:
            content = ""
        
        self._memory_cache[tenant_id] = content
        return content
    
    def _load_user(self, tenant_id: str, user_id: str) -> str:
        cache_key = f"{tenant_id}:{user_id}"
        if cache_key in self._user_cache:
            return self._user_cache[cache_key]
        
        path = self._get_user_path(tenant_id, user_id)
        if path.exists():
            content = path.read_text(encoding="utf-8")
        else:
            content = ""
        
        self._user_cache[cache_key] = content
        return content
    
    def _save_memory(self, tenant_id: str, content: str) -> None:
        path = self._get_memory_path(tenant_id)
        path.write_text(content, encoding="utf-8")
        self._memory_cache[tenant_id] = content
    
    def _save_user(self, tenant_id: str, user_id: str, content: str) -> None:
        path = self._get_user_path(tenant_id, user_id)
        path.write_text(content, encoding="utf-8")
        cache_key = f"{tenant_id}:{user_id}"
        self._user_cache[cache_key] = content
    
    def _truncate(self, content: str, max_chars: int) -> str:
        if len(content) <= max_chars:
            return content
        
        truncated = content[:max_chars]
        last_newline = truncated.rfind("\n")
        if last_newline > max_chars * 0.8:
            truncated = truncated[:last_newline]
        
        return truncated + "\n[... truncated ...]"
    
    def _compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def compose_prompt_snapshot(
        self,
        context: RequestContext
    ) -> PromptMemorySnapshot:
        memory_content = self._load_memory(context.tenant_id)
        user_content = self._load_user(context.tenant_id, context.user_id)
        
        memory_content = self._truncate(memory_content, self._hot_memory_max)
        user_content = self._truncate(user_content, self._user_memory_max)
        
        combined = f"# SYSTEM MEMORY\n\n{memory_content}\n\n# USER PREFERENCES\n\n{user_content}"
        
        snapshot_hash = self._compute_hash(combined)
        
        cache_key = f"{context.tenant_id}:{context.user_id}"
        self._snapshot_hash[cache_key] = snapshot_hash
        
        return PromptMemorySnapshot(
            memory_text=combined,
            hash=snapshot_hash,
            frozen=True
        )
    
    async def persist_turn(
        self,
        context: RequestContext,
        turn: TurnRecord
    ) -> None:
        if self._archive_db is None:
            self._logger.warning("Archive DB not configured, turn not persisted")
            return
        
        try:
            message_id = hashlib.sha256(
                f"{context.session_id}:{turn.role}:{turn.ts}".encode()
            ).hexdigest()[:16]
            
            await self._archive_db.insert("messages", {
                "message_id": message_id,
                "session_id": context.session_id,
                "role": turn.role,
                "content": turn.content,
                "ts": turn.ts or datetime.now().isoformat(),
                "token_count": len(turn.content) // 4,
                "redacted": 0,
                "metadata_json": json.dumps({
                    "tool_calls": turn.tool_calls,
                    "trace_id": turn.trace_id
                }) if turn.tool_calls or turn.trace_id else None
            })
            
            self._logger.debug(f"Persisted turn: {message_id}")
            
        except Exception as e:
            self._logger.error(f"Failed to persist turn: {e}")
    
    async def update_prompt_memory(
        self,
        context: RequestContext,
        delta: MemoryDelta,
        require_hitl: bool = True
    ) -> None:
        if require_hitl:
            self._logger.info(f"Memory update requires HITL: {delta.key}")
            return
        
        if delta.source == "memory":
            current = self._load_memory(context.tenant_id)
            updated = f"{current}\n\n{delta.key}: {delta.value}"
            updated = self._truncate(updated, self._hot_memory_max * 2)
            self._save_memory(context.tenant_id, updated)
        elif delta.source == "user":
            current = self._load_user(context.tenant_id, context.user_id)
            updated = f"{current}\n\n{delta.key}: {delta.value}"
            updated = self._truncate(updated, self._user_memory_max * 2)
            self._save_user(context.tenant_id, context.user_id, updated)
        else:
            self._logger.warning(f"Unknown memory source: {delta.source}")
    
    async def session_search(
        self,
        query: str,
        context: RequestContext,
        limit: int = 5
    ) -> str:
        if self._archive_db is None:
            return "Archive search not available"
        
        try:
            messages = await self._archive_db.search_messages(
                session_id=context.session_id,
                query=query,
                limit=limit * 3
            )
            
            if not messages:
                return "No relevant messages found"
            
            relevant = []
            query_lower = query.lower()
            
            for msg in messages:
                content = msg.get("content", "")
                if query_lower in content.lower():
                    role = msg.get("role", "user")
                    ts = msg.get("ts", "")
                    relevant.append(f"[{ts}] {role}: {content[:200]}...")
            
            if not relevant:
                return "No relevant messages found"
            
            return "\n\n".join(relevant[:limit])
            
        except Exception as e:
            self._logger.error(f"Session search failed: {e}")
            return f"Search error: {e}"
    
    async def skill_search(
        self,
        query: str,
        context: RequestContext,
        limit: int = 3
    ) -> List[SkillSummary]:
        skills_dir = Path("skills/published")
        if not skills_dir.exists():
            return []
        
        results = []
        query_lower = query.lower()
        
        for skill_file in skills_dir.glob("*/skill.yaml"):
            try:
                import yaml
                with open(skill_file, "r", encoding="utf-8") as f:
                    skill_data = yaml.safe_load(f) or {}
                
                title = skill_data.get("title", skill_file.parent.name)
                triggers = skill_data.get("triggers", [])
                
                score = 0
                if query_lower in title.lower():
                    score += 2
                for trigger in triggers:
                    if query_lower in trigger.lower():
                        score += 1
                
                if score > 0:
                    summary = f"{title}\nTriggers: {', '.join(triggers[:3])}"
                    results.append(SkillSummary(
                        skill_id=skill_file.parent.name,
                        title=title,
                        summary=summary
                    ))
                    
            except Exception as e:
                self._logger.debug(f"Failed to load skill {skill_file}: {e}")
        
        return results[:limit]
    
    def get_snapshot_hash(self, tenant_id: str, user_id: str) -> Optional[str]:
        cache_key = f"{tenant_id}:{user_id}"
        return self._snapshot_hash.get(cache_key)
    
    def invalidate_cache(self, tenant_id: str, user_id: Optional[str] = None) -> None:
        self._memory_cache.pop(tenant_id, None)
        
        if user_id:
            cache_key = f"{tenant_id}:{user_id}"
            self._user_cache.pop(cache_key, None)
        else:
            keys_to_remove = [k for k in self._user_cache if k.startswith(f"{tenant_id}:")]
            for k in keys_to_remove:
                del self._user_cache[k]
    
    def health(self) -> dict:
        return {
            "status": "healthy",
            "store_dir": str(self._store_dir),
            "archive_db": "configured" if self._archive_db else "not_configured",
            "cached_tenants": len(self._memory_cache),
            "cached_users": len(self._user_cache)
        }
