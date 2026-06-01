import hashlib
import json
import logging
from datetime import datetime
from typing import Any, List, Literal, Optional

from core.domain.context import RequestContext
from core.ports.external_memory import ExternalMemoryProvider
from core.ports.memory import (
    MemoryDelta,
    PromptMemorySnapshot,
    SkillOutcome,
    SkillSummary,
    ToolCallRecord,
    TurnRecord,
)
from core.ports.privacy import PrivacyPort
from core.ports.skills import SkillExecutionResult

from agent_platform.memory.adapters.hot_memory_compressor_adapter import (
    TruncatingHotMemoryCompressorAdapter,
)
from agent_platform.memory.adapters.hot_memory_file_adapter import HotMemoryFileAdapter
from agent_platform.memory.adapters.noop_external_adapter import (
    NoOpExternalMemoryAdapter,
)
from agent_platform.memory.adapters.skill_memory_adapter import SkillMemoryAdapter
from agent_platform.memory.adapters.summarizer_adapter import TruncatingSummarizerAdapter
from agent_platform.memory.adapters.session_search_fusion import rrf_merge_messages

SearchScope = Literal["session", "user"]


class MemoryPortAdapter:
    """
    MemoryPort 门面：组合 L1 热记忆、L2 冷档案、L3 技能、L4 外部画像。
    各层通过 Adapter 注入，符合 Port + Adapter 模式。
    """

    def __init__(
        self,
        store_dir: str = "workspace/memory",
        archive_db: Any = None,
        hot_memory_max_chars: int = 2200,
        user_memory_max_chars: int = 1375,
        session_search_cache_ttl: int = 900,
        retention_days: int = 90,
        hot_memory: Optional[HotMemoryFileAdapter] = None,
        privacy: Optional[PrivacyPort] = None,
        skill_memory: Optional[SkillMemoryAdapter] = None,
        summarizer: Any = None,
        compressor: Any = None,
        external_memory: Optional[ExternalMemoryProvider] = None,
        cache: Any = None,
        models: Any = None,
        session_vector_index: Any = None,
        session_hybrid_search: bool = True,
        object_store: Any = None,
        enable_cold_archive: bool = False,
        cold_archive_prefix: str = "l2/cold",
        cold_archive_compress: bool = True,
    ):
        self._archive_db = archive_db
        self._privacy = privacy
        self._skill_memory = skill_memory
        self._summarizer = summarizer or TruncatingSummarizerAdapter()
        self._compressor = compressor or TruncatingHotMemoryCompressorAdapter()
        self._external = external_memory or NoOpExternalMemoryAdapter()
        self._cache = cache
        self._models = models
        self._vector_index = session_vector_index
        self._session_hybrid_search = session_hybrid_search
        self._session_search_cache_ttl = session_search_cache_ttl
        self._retention_days = retention_days
        self._hot_max = hot_memory_max_chars
        self._user_max = user_memory_max_chars

        self._hot = hot_memory or HotMemoryFileAdapter(
            store_dir=store_dir,
            hot_memory_max_chars=hot_memory_max_chars,
            user_memory_max_chars=user_memory_max_chars,
        )
        self._cold_archive = None
        if enable_cold_archive and object_store is not None and archive_db is not None:
            from agent_platform.memory.adapters.session_cold_archive_service import (
                SessionColdArchiveService,
            )

            self._cold_archive = SessionColdArchiveService(
                archive_db,
                object_store,
                prefix=cold_archive_prefix,
                compress=cold_archive_compress,
            )
        self._logger = logging.getLogger(__name__)

    def compose_prompt_snapshot(
        self, context: RequestContext
    ) -> PromptMemorySnapshot:
        return self._hot.compose_snapshot(context)

    async def ensure_session(self, context: RequestContext) -> None:
        if self._archive_db is None:
            return
        await self._archive_db.upsert_session(
            {
                "session_id": context.session_id,
                "user_id": context.user_id,
                "tenant_id": context.tenant_id,
                "channel": context.channel,
                "started_at": datetime.now().isoformat(),
                "status": "active",
            }
        )

    async def finalize_session(self, context: RequestContext) -> None:
        pending = self._hot.flush_pending_deltas(
            context.tenant_id, context.user_id
        )
        for delta in pending:
            await self.apply_memory_delta(context, delta)

        facts = await self._external.fetch_profile_facts(
            context.user_id, context.tenant_id
        )
        for fact in facts:
            await self.apply_memory_delta(
                context,
                MemoryDelta(key=fact.key, value=fact.value, source="user"),
            )

        memory_raw = self._hot.get_raw_memory(context.tenant_id)
        if len(memory_raw) > self._hot_max:
            compressed = await self._compressor.compress_memory(
                memory_raw, self._hot_max
            )
            self._hot.save_memory(context.tenant_id, compressed)

        user_raw = self._hot.get_raw_user(context.tenant_id, context.user_id)
        if len(user_raw) > self._user_max:
            compressed = await self._compressor.compress_user(
                user_raw, self._user_max
            )
            self._hot.save_user(context.tenant_id, context.user_id, compressed)

        self._hot.invalidate_cache(context.tenant_id, context.user_id)
        await self._invalidate_session_search_cache(context)

    async def end_session(
        self,
        context: RequestContext,
        status: str = "closed",
        finalize: bool = True,
    ) -> None:
        if finalize:
            await self.finalize_session(context)
        if self._archive_db is None:
            return
        await self._archive_db.end_session(context.session_id, status=status)

    async def confirm_pending_deltas(self, context: RequestContext) -> int:
        pending = self._hot.flush_pending_deltas(
            context.tenant_id, context.user_id
        )
        for delta in pending:
            await self.apply_memory_delta(context, delta)
        return len(pending)

    @staticmethod
    def _message_id(context: RequestContext, turn: TurnRecord) -> str:
        ts = turn.ts or datetime.now().isoformat()
        prefix = (turn.content or "")[:64]
        raw = f"{context.session_id}:{turn.role}:{ts}:{prefix}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    async def persist_turn(
        self, context: RequestContext, turn: TurnRecord
    ) -> None:
        if self._archive_db is None:
            self._logger.warning("Archive DB not configured, turn not persisted")
            return

        await self.ensure_session(context)
        ts = turn.ts or datetime.now().isoformat()
        record = {
            "message_id": self._message_id(context, turn),
            "session_id": context.session_id,
            "role": turn.role,
            "content": turn.content,
            "ts": ts,
            "token_count": len(turn.content or "") // 4,
            "redacted": False,
            "metadata_json": json.dumps(
                {"tool_calls": turn.tool_calls, "trace_id": turn.trace_id},
                ensure_ascii=False,
            )
            if turn.tool_calls or turn.trace_id
            else None,
        }
        if self._privacy:
            record = self._privacy.redact_for_storage(record)

        await self._archive_db.insert_message(record)
        if self._vector_index is not None:
            try:
                await self._vector_index.index_message(
                    {
                        **record,
                        "tenant_id": context.tenant_id,
                        "user_id": context.user_id,
                    }
                )
            except Exception as e:
                self._logger.warning("Session vector index failed: %s", e)
        await self._invalidate_session_search_cache(context)

    async def persist_tool_call(
        self, context: RequestContext, record: ToolCallRecord
    ) -> None:
        if self._archive_db is None:
            return

        await self.ensure_session(context)
        ts = record.ts or datetime.now().isoformat()
        call_id = hashlib.sha256(
            f"{context.session_id}:{record.tool_name}:{ts}".encode()
        ).hexdigest()[:16]
        row = {
            "call_id": call_id,
            "session_id": context.session_id,
            "tool_name": record.tool_name,
            "args_hash": record.args_hash,
            "result_summary": record.result_summary,
            "status": record.status,
            "latency_ms": record.latency_ms,
            "ts": ts,
        }
        if self._privacy:
            row = self._privacy.redact_for_storage(row)
        await self._archive_db.insert_tool_call(row)

    async def list_turns(
        self, context: RequestContext, limit: int = 100, offset: int = 0
    ) -> List[dict]:
        if self._archive_db is None:
            return []
        return await self._archive_db.select_many(
            "messages",
            [
                "message_id",
                "session_id",
                "role",
                "content",
                "ts",
                "token_count",
                "metadata_json",
            ],
            where={"session_id": context.session_id},
            order_by="ts ASC",
            limit=limit,
            offset=offset or None,
        )

    async def list_sessions(
        self, tenant_id: str, user_id: str, limit: int = 20
    ) -> List[dict]:
        if self._archive_db is None:
            return []
        return await self._archive_db.list_sessions(tenant_id, user_id, limit=limit)

    async def apply_memory_delta(
        self, context: RequestContext, delta: MemoryDelta
    ) -> None:
        self._hot.apply_delta(context.tenant_id, context.user_id, delta)
        self._hot.invalidate_cache(context.tenant_id, context.user_id)

    async def update_prompt_memory(
        self,
        context: RequestContext,
        delta: MemoryDelta,
        require_hitl: bool = True,
    ) -> None:
        if require_hitl:
            self._hot.queue_pending_delta(
                context.tenant_id, context.user_id, delta
            )
            return
        await self.apply_memory_delta(context, delta)

    async def _cache_get(self, key: str) -> Optional[str]:
        if self._cache is None:
            return None
        getter = getattr(self._cache, "get", None)
        if getter is None:
            return None
        value = getter(key)
        if hasattr(value, "__await__"):
            value = await value
        return value

    async def _cache_set(self, key: str, value: str) -> None:
        if self._cache is None:
            return
        setter = getattr(self._cache, "set", None)
        if setter is None:
            return
        result = setter(key, value, ttl_seconds=self._session_search_cache_ttl)
        if hasattr(result, "__await__"):
            await result

    def _session_search_cache_key(
        self,
        context: RequestContext,
        query: str,
        limit: int,
        scope: SearchScope,
    ) -> str:
        raw = (
            f"{context.tenant_id}:{context.user_id}:"
            f"{context.session_id}:{scope}:{query}:{limit}"
        )
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        if hasattr(self._cache, "build_key"):
            return self._cache.build_key(context.tenant_id, "sess", digest)
        return f"sess:{context.tenant_id}:{digest}"

    async def _invalidate_session_search_cache(
        self, context: RequestContext
    ) -> None:
        if self._cache is None:
            return
        invalidator = getattr(self._cache, "invalidate_pattern", None)
        if invalidator is None:
            return
        pattern = f"*{context.tenant_id}*sess*"
        result = invalidator(pattern)
        if hasattr(result, "__await__"):
            await result

    def _filter_messages_by_acl(
        self, messages: List[dict], context: RequestContext
    ) -> List[dict]:
        filtered: List[dict] = []
        for msg in messages:
            if not msg.get("session_id"):
                continue
            if msg.get("redacted"):
                continue
            content = (msg.get("content") or "").strip()
            if not content or content == "[redacted]":
                continue
            filtered.append(msg)
        return filtered

    @staticmethod
    def _format_search_fragment(msg: dict, scope: SearchScope) -> str:
        role = msg.get("role", "user")
        ts = msg.get("ts", "")
        content = (msg.get("content") or "").strip()
        preview = content if len(content) <= 200 else content[:200] + "..."
        if scope == "user":
            session_id = msg.get("session_id", "")
            return f"[session={session_id}] [{ts}] {role}: {preview}"
        return f"[{ts}] {role}: {preview}"

    async def _search_archive_messages(
        self,
        query: str,
        context: RequestContext,
        *,
        session_id: Optional[str],
        limit: int,
    ) -> List[dict]:
        fts_messages = await self._archive_db.search_messages(
            session_id=session_id,
            user_id=context.user_id,
            tenant_id=context.tenant_id,
            query=query,
            limit=limit,
        )
        if self._vector_index is None:
            return fts_messages

        try:
            vector_messages = await self._vector_index.search(
                query,
                context.tenant_id,
                context.user_id,
                session_id=session_id,
                limit=limit,
            )
        except Exception as e:
            self._logger.warning("Session vector search failed: %s", e)
            return fts_messages

        if not vector_messages:
            return fts_messages
        if not fts_messages:
            return vector_messages
        if self._session_hybrid_search:
            return rrf_merge_messages([fts_messages, vector_messages])[:limit]
        return fts_messages

    async def session_search(
        self,
        query: str,
        context: RequestContext,
        limit: int = 5,
        scope: SearchScope = "session",
    ) -> str:
        if self._archive_db is None:
            return "Archive search not available"

        cache_key = self._session_search_cache_key(context, query, limit, scope)
        cached = await self._cache_get(cache_key)
        if cached:
            return cached

        session_id = context.session_id if scope == "session" else None

        try:
            messages = await self._search_archive_messages(
                query,
                context,
                session_id=session_id,
                limit=limit * 3,
            )
            messages = self._filter_messages_by_acl(messages, context)
            if not messages:
                return "No relevant messages found"

            fragments = [
                self._format_search_fragment(msg, scope)
                for msg in messages[: limit * 3]
            ]

            if not fragments:
                return "No relevant messages found"

            result = await self._summarizer.summarize(fragments[:limit], query)
            await self._cache_set(cache_key, result)
            return result
        except Exception as e:
            self._logger.error("Session search failed: %s", e)
            return f"Search error: {e}"

    async def reindex_session_vectors(
        self,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        batch_size: int = 200,
    ) -> dict:
        if self._archive_db is None:
            return {
                "indexed": 0,
                "errors": 0,
                "batches": 0,
                "reason": "archive_db_not_configured",
            }
        if self._vector_index is None:
            return {
                "indexed": 0,
                "errors": 0,
                "batches": 0,
                "reason": "session_vector_index_not_configured",
            }

        lister = getattr(self._archive_db, "list_messages_for_reindex", None)
        if lister is None:
            return {
                "indexed": 0,
                "errors": 0,
                "batches": 0,
                "reason": "list_messages_for_reindex_not_supported",
            }

        indexed = 0
        errors = 0
        batches = 0
        offset = 0

        while True:
            batch = await lister(
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                limit=batch_size,
                offset=offset,
            )
            if not batch:
                break

            batches += 1
            for row in batch:
                try:
                    await self._vector_index.index_message(row)
                    indexed += 1
                except Exception as e:
                    errors += 1
                    self._logger.warning(
                        "Reindex failed message_id=%s: %s",
                        row.get("message_id"),
                        e,
                    )

            offset += len(batch)
            if len(batch) < batch_size:
                break

        if tenant_id and user_id:
            await self._invalidate_session_search_cache(
                RequestContext(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    session_id=session_id or "reindex",
                    trace_id="reindex",
                    channel="system",
                )
            )

        return {"indexed": indexed, "errors": errors, "batches": batches}

    async def skill_search(
        self,
        query: str,
        context: RequestContext,
        limit: int = 3,
    ) -> List[SkillSummary]:
        if self._skill_memory is None:
            return []
        return self._skill_memory.search(query, context.tenant_id, limit=limit)

    async def run_skill(
        self,
        skill_id: str,
        inputs: dict,
        context: RequestContext,
        run_context: Any,
    ) -> SkillExecutionResult:
        if self._skill_memory is None:
            return SkillExecutionResult(
                skill_id=skill_id,
                success=False,
                error="SkillMemoryAdapter not configured",
            )
        result = self._skill_memory.run(skill_id, inputs, run_context)
        await self.record_skill_outcome(
            context,
            SkillOutcome(
                skill_id=skill_id,
                success=result.success,
                steps_executed=result.steps_executed,
                error=result.error,
            ),
        )
        return result

    async def record_skill_outcome(
        self, context: RequestContext, outcome: SkillOutcome
    ) -> None:
        if self._skill_memory is None:
            return
        self._skill_memory.record_outcome(context.tenant_id, outcome)

    async def resolve_entity(
        self, mention: str, context: RequestContext
    ) -> Optional[Any]:
        return await self._external.resolve_entity(mention, context)

    async def purge_user_data(self, tenant_id: str, user_id: str) -> None:
        self._hot.clear_user(tenant_id, user_id)
        if self._archive_db is not None:
            await self._archive_db.anonymize_user_data(tenant_id, user_id)
        if self._cold_archive is not None:
            try:
                await self._cold_archive.delete_cold_archives_for_user(
                    tenant_id, user_id
                )
            except Exception as e:
                self._logger.warning("Cold archive purge user failed: %s", e)
        if self._vector_index is not None:
            try:
                await self._vector_index.delete_user_messages(tenant_id, user_id)
            except Exception as e:
                self._logger.warning("Session vector purge user failed: %s", e)
        ctx = RequestContext(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id="purge",
            trace_id="purge",
            channel="system",
        )
        await self._invalidate_session_search_cache(ctx)

    async def archive_expired_sessions(
        self, retention_days: Optional[int] = None
    ) -> dict:
        if self._cold_archive is None:
            return {
                "archived": 0,
                "candidates": 0,
                "errors": 0,
                "reason": "cold_archive_not_configured",
            }
        days = retention_days if retention_days is not None else self._retention_days
        if self._vector_index is not None:
            try:
                expired_ids = await self._archive_db.list_expired_session_ids(days)
                for session_id in expired_ids:
                    await self._vector_index.delete_session_messages(session_id)
            except Exception as e:
                self._logger.warning(
                    "Session vector cleanup before cold archive failed: %s", e
                )
        return await self._cold_archive.archive_expired_sessions(days)

    async def archive_session(self, session_id: str) -> dict:
        if self._cold_archive is None:
            return {"reason": "cold_archive_not_configured"}
        if self._vector_index is not None:
            try:
                await self._vector_index.delete_session_messages(session_id)
            except Exception as e:
                self._logger.warning("Session vector cleanup failed: %s", e)
        return await self._cold_archive.archive_session(session_id)

    async def list_cold_archives(
        self, tenant_id: str, user_id: str, limit: int = 20
    ) -> List[dict]:
        if self._cold_archive is None:
            return []
        return await self._cold_archive.list_cold_archives(tenant_id, user_id, limit)

    async def fetch_cold_session(self, session_id: str) -> Optional[dict]:
        if self._cold_archive is None:
            return None
        return await self._cold_archive.fetch_cold_session(session_id)

    async def purge_expired_sessions(self, retention_days: Optional[int] = None) -> int:
        if self._archive_db is None:
            return 0
        days = retention_days if retention_days is not None else self._retention_days
        if self._cold_archive is not None:
            result = await self.archive_expired_sessions(days)
            return int(result.get("archived", 0))
        if self._vector_index is not None:
            try:
                expired_ids = await self._archive_db.list_expired_session_ids(days)
                for session_id in expired_ids:
                    await self._vector_index.delete_session_messages(session_id)
            except Exception as e:
                self._logger.warning("Session vector purge expired failed: %s", e)
        return await self._archive_db.purge_expired_sessions(days)

    def get_snapshot_hash(self, tenant_id: str, user_id: str) -> Optional[str]:
        return self._hot.get_snapshot_hash(tenant_id, user_id)

    def invalidate_cache(
        self, tenant_id: str, user_id: Optional[str] = None
    ) -> None:
        self._hot.invalidate_cache(tenant_id, user_id)

    def health(self) -> dict:
        return {
            "status": "healthy",
            "store_dir": str(self._hot.store_dir),
            "archive_db": "configured" if self._archive_db else "not_configured",
            "skill_memory": "configured" if self._skill_memory else "not_configured",
            "external_memory": type(self._external).__name__,
            "summarizer": type(self._summarizer).__name__,
            "compressor": type(self._compressor).__name__,
            "cache": "configured" if self._cache else "not_configured",
            "models": "configured" if self._models else "not_configured",
            "session_vector_index": (
                "configured" if self._vector_index else "not_configured"
            ),
            "session_hybrid_search": self._session_hybrid_search,
            "cold_archive": (
                "configured" if self._cold_archive else "not_configured"
            ),
        }
