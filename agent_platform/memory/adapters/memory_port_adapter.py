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
    SessionFragment,
    SessionSearchResult,
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
from agent_platform.memory.adapters.session_rerank_utils import rerank_message_dicts
from app.agents.prompts.text_sanitize import (
    has_model_reasoning,
    sanitize_search_fragment_content,
    strip_model_reasoning,
)

SearchScope = Literal["session", "user"]


SESSION_SEARCH_EMPTY_SENTINEL = "__SESSION_SEARCH_EMPTY__"


def _estimate_token_count(text: str) -> int:
    try:
        from utils.token_counter import count_tokens

        return count_tokens(text or "")
    except Exception:
        return max(1, len(text or "") // 4)


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
        session_search_negative_cache_ttl: int = 120,
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
        session_search_cold_fallback: bool = True,
        session_search_rerank: bool = True,
        cold_archive_search_scan_limit: int = 100,
        cold_archive_keep_vectors: bool = True,
        session_vector_auto_reindex: bool = True,
        reindex_batch_size: int = 200,
        index_port: Any = None,
        cold_archive_encrypt_at_rest: bool = False,
        encryption_key: Optional[str] = None,
        external_merge_on_finalize: bool = True,
        purge_delete_external_audit: bool = True,
        purge_tenant_l4_strip_user_keys: bool = True,
        time_decay: bool = True,
        time_decay_half_life_days: float = 90.0,
    ):
        self._archive_db = archive_db
        self._index_port = index_port
        self._privacy = privacy
        self._skill_memory = skill_memory
        self._summarizer = summarizer or TruncatingSummarizerAdapter()
        self._truncate_summarizer = TruncatingSummarizerAdapter(max_chars=2000)
        self._compressor = compressor or TruncatingHotMemoryCompressorAdapter()
        self._external = external_memory or NoOpExternalMemoryAdapter()
        self._cache = cache
        self._models = models
        self._vector_index = session_vector_index
        self._session_hybrid_search = session_hybrid_search
        self._session_search_cache_ttl = session_search_cache_ttl
        self._session_search_negative_cache_ttl = session_search_negative_cache_ttl
        self._session_search_cold_fallback = session_search_cold_fallback
        self._session_search_rerank = session_search_rerank
        self._cold_archive_keep_vectors = cold_archive_keep_vectors
        self._session_vector_auto_reindex = session_vector_auto_reindex
        self._reindex_batch_size = reindex_batch_size
        self._vector_version_checked = False
        self._retention_days = retention_days
        self._hot_max = hot_memory_max_chars
        self._user_max = user_memory_max_chars
        self._external_merge_on_finalize = external_merge_on_finalize
        self._purge_delete_external_audit = purge_delete_external_audit
        self._purge_tenant_l4_strip_user_keys = purge_tenant_l4_strip_user_keys
        self._time_decay_enabled = time_decay
        self._time_decay_half_life_days = time_decay_half_life_days

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
                search_scan_limit=cold_archive_search_scan_limit,
                encrypt_at_rest=cold_archive_encrypt_at_rest,
                encryption_key=encryption_key,
            )
        self._logger = logging.getLogger(__name__)

    async def _ensure_session_vector_index_version(self) -> None:
        if self._vector_index is None or self._vector_version_checked:
            return

        self._vector_version_checked = True
        is_current = getattr(self._vector_index, "is_version_current", None)
        if is_current is None or is_current():
            return

        stored = getattr(self._vector_index, "get_stored_version", lambda: "?")()
        expected = getattr(self._vector_index, "index_version", "?")
        count_fn = getattr(getattr(self._vector_index, "_vector", None), "count", None)
        collection = getattr(self._vector_index, "_collection", "")
        if count_fn is not None and count_fn(collection) == 0:
            marker = getattr(self._vector_index, "mark_index_current", None)
            if marker:
                marker()
            return

        if not self._session_vector_auto_reindex:
            self._logger.warning(
                "Session vector index version mismatch stored=%s expected=%s; "
                "run reindex or set session_vector_auto_reindex_on_version_change",
                stored,
                expected,
            )
            return

        self._logger.info(
            "Session vector index version mismatch stored=%s expected=%s; reindexing",
            stored,
            expected,
        )
        await self.reindex_session_vectors(batch_size=self._reindex_batch_size)

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

    async def finalize_session(self, context: RequestContext) -> dict:
        pending = self._hot.flush_pending_deltas(
            context.tenant_id, context.user_id
        )
        for delta in pending:
            await self.apply_memory_delta(context, delta)

        l4_changes: List[dict] = []
        if self._external_merge_on_finalize:
            l4_changes = await self._merge_l4_facts_into_user(context)

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
        await self._invalidate_session_search_cache(context, reason="finalize")
        return {
            "pending_applied": len(pending),
            "l4_merged": len(l4_changes),
            "l4_keys": [str(c.get("key") or "") for c in l4_changes if c.get("key")],
        }

    async def refresh_external_profile(
        self, context: RequestContext
    ) -> dict:
        """失效 L4 缓存并重新拉取 facts（不写入 L1）。"""
        ext = self._external
        invalidator = getattr(ext, "invalidate_user_profile_cache", None)
        if callable(invalidator):
            await invalidator(context.tenant_id, context.user_id)
        elif hasattr(ext, "_invalidate_user"):
            await ext._invalidate_user(context.tenant_id, context.user_id)
        facts = await self.fetch_profile_facts(
            context.tenant_id, context.user_id
        )
        return {
            "refreshed": True,
            "fact_count": len(facts),
            "facts": facts[:12],
        }

    async def end_session(
        self,
        context: RequestContext,
        status: str = "closed",
        finalize: bool = True,
    ) -> dict:
        summary: dict = {}
        if finalize:
            summary = await self.finalize_session(context)
        if self._archive_db is None:
            return summary
        await self._archive_db.end_session(context.session_id, status=status)
        return summary

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

        await self._ensure_session_vector_index_version()
        await self.ensure_session(context)
        ts = turn.ts or datetime.now().isoformat()
        record = {
            "message_id": self._message_id(context, turn),
            "session_id": context.session_id,
            "role": turn.role,
            "content": turn.content,
            "ts": ts,
            "token_count": _estimate_token_count(turn.content or ""),
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
        if self._vector_index is not None:
            indexer = getattr(self._vector_index, "index_tool_call", None)
            if indexer is not None:
                try:
                    await indexer(
                        {
                            **row,
                            "tenant_id": context.tenant_id,
                            "user_id": context.user_id,
                        }
                    )
                except Exception as e:
                    self._logger.warning("Session vector tool_call index failed: %s", e)

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
        from app.agents.context_builder import is_allowed_l1_value

        if delta.source in ("user", "external"):
            if not is_allowed_l1_value(delta.key, delta.value):
                self._logger.warning(
                    "Rejected L1 delta key=%s (value not allowed)", delta.key
                )
                return
            changes = self._hot.merge_user_facts_upsert(
                context.tenant_id, context.user_id, [delta]
            )
            await self._audit_l1_changes(context, changes)
        elif delta.source == "memory":
            self._hot.apply_delta(context.tenant_id, context.user_id, delta)
        else:
            self._hot.apply_delta(context.tenant_id, context.user_id, delta)
        self._hot.invalidate_cache(context.tenant_id, context.user_id)

    async def _audit_l1_changes(
        self, context: RequestContext, changes: List[dict]
    ) -> None:
        if not changes or self._archive_db is None:
            return
        from agent_platform.memory.adapters.compliance_utils import (
            append_audit_log,
            content_sha256,
        )

        for change in changes:
            await append_audit_log(
                self._archive_db,
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                resource_type="l1_user",
                resource_id=str(change.get("key") or ""),
                content_hash=content_sha256(str(change.get("new") or "")),
                action="upsert",
                meta=change,
            )

    async def update_prompt_memory(
        self,
        context: RequestContext,
        delta: MemoryDelta,
        require_hitl: bool = True,
    ) -> None:
        from app.agents.context_builder import is_allowed_l1_value

        if not is_allowed_l1_value(delta.key, delta.value):
            self._logger.warning(
                "Rejected pending L1 key=%s (value not allowed)", delta.key
            )
            return
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

    def _cache_adapter_label(self) -> str:
        if self._cache is None:
            return "none"
        return type(self._cache).__name__

    async def _cache_set(
        self, key: str, value: str, *, ttl_seconds: Optional[int] = None
    ) -> None:
        if self._cache is None:
            return
        setter = getattr(self._cache, "set", None)
        if setter is None:
            return
        ttl = (
            self._session_search_cache_ttl
            if ttl_seconds is None
            else ttl_seconds
        )
        result = setter(key, value, ttl_seconds=ttl)
        if hasattr(result, "__await__"):
            await result

    def _session_search_cache_key(
        self,
        context: RequestContext,
        query: str,
        limit: int,
        scope: SearchScope,
        *,
        use_llm_summary: bool | None = None,
        prefer_user_role: bool = False,
    ) -> str:
        if use_llm_summary is not False:
            mode = "llm"
        elif prefer_user_role:
            mode = "truncate_user"
        else:
            mode = "truncate"
        raw = (
            f"{context.tenant_id}:{context.user_id}:"
            f"{context.session_id}:{scope}:{query}:{limit}:{mode}"
        )
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        if hasattr(self._cache, "build_key"):
            return self._cache.build_key(context.tenant_id, "sess", digest)
        return f"sess:{context.tenant_id}:{digest}"

    async def _invalidate_session_search_cache(
        self, context: RequestContext, *, reason: str = "unknown"
    ) -> None:
        if self._cache is None:
            return
        invalidator = getattr(self._cache, "invalidate_pattern", None)
        if invalidator is None:
            return
        pattern = f"*{context.tenant_id}:*:sess:*"
        result = invalidator(pattern)
        if hasattr(result, "__await__"):
            result = await result

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

    def _pick_summarizer(self, use_llm_summary: bool | None):
        if use_llm_summary is False:
            return self._truncate_summarizer
        return self._summarizer

    @staticmethod
    def _format_search_fragment(msg: dict, scope: SearchScope) -> str:
        role = msg.get("role", "user")
        ts = msg.get("ts", "")
        content = sanitize_search_fragment_content(
            msg.get("content"), role=str(role)
        )
        if not content:
            return ""
        preview = content if len(content) <= 200 else content[:200] + "..."
        if scope == "user":
            session_id = msg.get("session_id", "")
            return f"[session={session_id}] [{ts}] {role}: {preview}"
        return f"[{ts}] {role}: {preview}"

    def _mask_search_text(self, text: str) -> str:
        if self._privacy is None:
            return text
        masker = getattr(self._privacy, "mask_text", None)
        if masker is None:
            return text
        return masker(text)

    async def _search_tool_calls(
        self,
        query: str,
        context: RequestContext,
        *,
        session_id: Optional[str],
        limit: int,
    ) -> List[dict]:
        searcher = getattr(self._archive_db, "search_tool_calls", None)
        if searcher is None:
            return []
        try:
            return await searcher(
                session_id=session_id,
                user_id=context.user_id,
                tenant_id=context.tenant_id,
                query=query,
                limit=limit,
            )
        except Exception as e:
            self._logger.warning("Tool call search failed: %s", e)
            return []

    async def _search_cold_messages(
        self,
        query: str,
        context: RequestContext,
        *,
        session_id: Optional[str],
        limit: int,
    ) -> List[dict]:
        if self._cold_archive is None or not self._session_search_cold_fallback:
            return []
        try:
            return await self._cold_archive.search_cold_archives(
                query,
                context.tenant_id,
                context.user_id,
                session_id=session_id,
                limit=limit,
            )
        except Exception as e:
            self._logger.warning("Cold archive search failed: %s", e)
            return []

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
        tool_messages = await self._search_tool_calls(
            query, context, session_id=session_id, limit=limit
        )

        ranked_lists = []
        if fts_messages:
            ranked_lists.append(fts_messages)
        if tool_messages:
            ranked_lists.append(tool_messages)

        if self._vector_index is not None:
            try:
                vector_messages = await self._vector_index.search(
                    query,
                    context.tenant_id,
                    context.user_id,
                    session_id=session_id,
                    limit=limit,
                )
                if vector_messages:
                    ranked_lists.append(vector_messages)
            except Exception as e:
                self._logger.warning("Session vector search failed: %s", e)

        if not ranked_lists:
            online: List[dict] = []
        elif len(ranked_lists) == 1:
            online = ranked_lists[0]
        elif self._session_hybrid_search:
            online = rrf_merge_messages(ranked_lists)[:limit]
        else:
            online = ranked_lists[0][:limit]

        cold = await self._search_cold_messages(
            query, context, session_id=session_id, limit=limit
        )
        if not online:
            merged = cold
        elif not cold:
            merged = online
        else:
            merged = rrf_merge_messages([online, cold])[: limit * 2]

        if self._session_search_rerank and merged:
            merged = rerank_message_dicts(query, merged, top_n=limit * 2)
        return merged[: limit * 3]

    async def _search_user_messages_fallback(
        self,
        query: str,
        context: RequestContext,
        session_id: Optional[str],
        limit: int,
    ) -> List[dict]:
        """混合检索仅命中 assistant 时，回退到 user 发言关键词匹配。"""
        if self._archive_db is None:
            return []
        where: dict[str, str] = {}
        if session_id:
            where["session_id"] = session_id
        rows = await self._archive_db.select_many(
            "messages",
            ["message_id", "session_id", "role", "content", "ts"],
            where=where or None,
            order_by="ts DESC",
            limit=min(300, limit * 40),
        )
        user_rows = [r for r in rows if str(r.get("role") or "") == "user"]
        q = (query or "").strip()
        tokens = [t for t in q.split() if len(t) >= 2]
        if not tokens:
            return user_rows[:limit]
        scored: List[tuple[int, dict]] = []
        for row in user_rows:
            content = str(row.get("content") or "")
            score = sum(1 for t in tokens if t in content)
            if score > 0:
                scored.append((score, row))
        if not scored:
            return []
        scored.sort(key=lambda x: (-x[0], str(x[1].get("ts") or "")))
        return [row for _, row in scored[:limit]]

    def _messages_to_fragments(
        self, messages: List[dict], scope: SearchScope
    ) -> List[SessionFragment]:
        fragments: List[SessionFragment] = []
        for msg in messages:
            role = str(msg.get("role") or "user")
            content = sanitize_search_fragment_content(
                msg.get("content"), role=role
            )
            if not content:
                continue
            fragments.append(
                SessionFragment(
                    message_id=str(msg.get("message_id") or ""),
                    session_id=str(msg.get("session_id") or ""),
                    role=str(msg.get("role") or "user"),
                    content=content,
                    ts=str(msg.get("ts") or ""),
                    score=float(msg.get("rerank_score") or msg.get("score") or 0.0),
                    source=str(msg.get("source") or "online"),
                )
            )
        return fragments

    async def session_search_detail(
        self,
        query: str,
        context: RequestContext,
        limit: int = 5,
        scope: SearchScope = "session",
        *,
        use_llm_summary: bool | None = None,
        prefer_user_role: bool = False,
    ) -> SessionSearchResult:
        if self._archive_db is None:
            return SessionSearchResult(
                summary="Archive search not available", sources=[]
            )

        await self._ensure_session_vector_index_version()
        session_id = context.session_id if scope == "session" else None
        messages = await self._search_archive_messages(
            query,
            context,
            session_id=session_id,
            limit=limit * 3,
        )
        messages = self._filter_messages_by_acl(messages, context)
        if prefer_user_role:
            user_msgs = [
                m for m in messages if str(m.get("role") or "") == "user"
            ]
            if user_msgs:
                messages = user_msgs
            else:
                fallback = await self._search_user_messages_fallback(
                    query, context, session_id=session_id, limit=limit * 3
                )
                if fallback:
                    messages = fallback
        if not messages:
            return SessionSearchResult(
                summary="No relevant messages found", sources=[]
            )

        fragments = self._messages_to_fragments(messages[: limit * 3], scope)
        if not fragments:
            return SessionSearchResult(
                summary="No relevant messages found", sources=[]
            )

        selected = fragments[:limit]
        # 时间衰减：近期结果权重高于远期
        if self._time_decay_enabled:
            from agent_platform.memory.adapters.time_decay import (
                apply_time_decay_to_fragments,
            )

            frag_dicts = [
                {"ts": f.ts, "score": f.score, "idx": i}
                for i, f in enumerate(selected)
            ]
            decayed = apply_time_decay_to_fragments(
                frag_dicts,
                half_life_days=self._time_decay_half_life_days,
            )
            # 按衰减后分数重排
            idx_order = [d["idx"] for d in decayed]
            selected = [selected[i] for i in idx_order if i < len(selected)]

        if use_llm_summary is False:
            user_frags = [f for f in fragments if f.role == "user"]
            if user_frags:
                selected = user_frags[:limit]
            else:
                # 知识预检索：无 user 命中时不回退 assistant 脏片段
                return SessionSearchResult(
                    summary="No relevant messages found", sources=[]
                )

        text_fragments = [
            frag
            for f in selected
            if (frag := self._format_search_fragment(
                {
                    "session_id": f.session_id,
                    "role": f.role,
                    "ts": f.ts,
                    "content": f.content,
                },
                scope,
            ))
        ]
        if not text_fragments:
            return SessionSearchResult(
                summary="No relevant messages found", sources=[]
            )

        summarizer = self._pick_summarizer(use_llm_summary)
        summary = await summarizer.summarize(text_fragments, query)
        summary = strip_model_reasoning(summary)
        if has_model_reasoning(summary):
            summary = await self._truncate_summarizer.summarize(
                text_fragments, query
            )
            summary = strip_model_reasoning(summary)
        summary = self._mask_search_text(summary)
        sources = sorted({f.source for f in selected})
        return SessionSearchResult(
            summary=summary,
            fragments=selected,
            sources=sources,
        )

    async def session_search(
        self,
        query: str,
        context: RequestContext,
        limit: int = 5,
        scope: SearchScope = "session",
        *,
        use_llm_summary: bool | None = None,
        prefer_user_role: bool = False,
    ) -> str:
        if self._archive_db is None:
            return "Archive search not available"

        cache_key = self._session_search_cache_key(
            context,
            query,
            limit,
            scope,
            use_llm_summary=use_llm_summary,
            prefer_user_role=prefer_user_role,
        )
        cached = await self._cache_get(cache_key)
        if cached is not None:
            if cached == SESSION_SEARCH_EMPTY_SENTINEL:
                return ""
            return cached

        try:
            detail = await self.session_search_detail(
                query,
                context,
                limit=limit,
                scope=scope,
                use_llm_summary=use_llm_summary,
                prefer_user_role=prefer_user_role,
            )
            if not detail.fragments:
                if self._session_search_negative_cache_ttl > 0:
                    await self._cache_set(
                        cache_key,
                        SESSION_SEARCH_EMPTY_SENTINEL,
                        ttl_seconds=self._session_search_negative_cache_ttl,
                    )
                return ""
            await self._cache_set(cache_key, detail.summary)
            return detail.summary
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
                ),
                reason="reindex",
            )

        marker = getattr(self._vector_index, "mark_index_current", None)
        if marker is not None and errors == 0:
            marker()

        version = getattr(self._vector_index, "index_version", None)
        return {
            "indexed": indexed,
            "errors": errors,
            "batches": batches,
            "index_version": version,
        }

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
        result = await self._skill_memory.run_and_finalize(
            skill_id, inputs, run_context, context
        )
        return result

    async def list_skills(self, context: RequestContext) -> List[dict]:
        if self._skill_memory is None:
            return []
        return self._skill_memory.list_skills(context.tenant_id)

    async def get_skill(
        self, skill_id: str, context: RequestContext
    ) -> Optional[dict]:
        if self._skill_memory is None:
            return None
        return self._skill_memory.get_skill_detail(skill_id, context.tenant_id)

    async def extract_skill_draft(
        self,
        context: RequestContext,
        title: str,
        triggers: List[str],
        steps: List[dict],
        skill_id: Optional[str] = None,
    ) -> str:
        if self._skill_memory is None:
            raise RuntimeError("SkillMemoryAdapter not configured")
        return self._skill_memory.extract_draft(
            context.tenant_id, title, triggers, steps, skill_id=skill_id
        )

    async def list_skill_drafts(self, context: RequestContext) -> List[dict]:
        if self._skill_memory is None:
            return []
        return self._skill_memory.list_drafts(context.tenant_id)

    async def publish_skill(
        self,
        context: RequestContext,
        skill_id: str,
        *,
        remove_draft: bool = True,
    ) -> dict:
        if self._skill_memory is None:
            return {"success": False, "reason": "skill_memory_not_configured"}
        return self._skill_memory.publish_skill(
            context.tenant_id, skill_id, remove_draft=remove_draft
        )

    async def deprecate_skill(
        self, context: RequestContext, skill_id: str
    ) -> dict:
        if self._skill_memory is None:
            return {"success": False, "reason": "skill_memory_not_configured"}
        return self._skill_memory.deprecate_skill(context.tenant_id, skill_id)

    async def activate_skill(
        self, context: RequestContext, skill_id: str
    ) -> dict:
        if self._skill_memory is None:
            return {"success": False, "reason": "skill_memory_not_configured"}
        return self._skill_memory.activate_skill(context.tenant_id, skill_id)

    async def sync_skills_from(
        self,
        source_dir: str,
        *,
        remove_missing: bool = False,
    ) -> dict:
        if self._skill_memory is None:
            return {"success": False, "reason": "skill_memory_not_configured"}
        return self._skill_memory.sync_from_source(
            source_dir, remove_missing=remove_missing
        )

    async def list_skill_runs(
        self,
        context: RequestContext,
        *,
        skill_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[dict]:
        if self._skill_memory is None:
            return []
        return await self._skill_memory.list_skill_runs(
            context.tenant_id,
            user_id=context.user_id,
            skill_id=skill_id,
            limit=limit,
            offset=offset,
        )

    async def purge_tenant_l3(
        self,
        tenant_id: str,
        *,
        delete_runs: bool = True,
    ) -> dict:
        if self._skill_memory is None:
            return {"success": False, "reason": "skill_memory_not_configured"}
        result = await self._skill_memory.purge_l3_for_tenant_async(
            tenant_id, delete_runs=delete_runs
        )
        return {"success": True, **result}

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

    async def fetch_profile_facts(
        self, tenant_id: str, user_id: str
    ) -> List[Any]:
        facts = await self._external.fetch_profile_facts(user_id, tenant_id)
        return [
            {"key": f.key, "value": f.value, "source": f.source}
            for f in facts
        ]

    async def list_profile_users(self, tenant_id: str) -> List[str]:
        return await self._external.list_profile_users(tenant_id)

    async def get_profile(self, tenant_id: str, user_id: str) -> dict:
        return await self._external.get_profile(tenant_id, user_id)

    async def import_profile(
        self, tenant_id: str, user_id: str, profile: dict
    ) -> None:
        await self._external.save_profile(tenant_id, user_id, profile)

    async def set_profile_facts(
        self, tenant_id: str, user_id: str, facts: List[dict]
    ) -> int:
        return await self._external.upsert_profile_facts(
            tenant_id, user_id, facts
        )

    async def purge_tenant_l4(self, tenant_id: str) -> dict:
        keys_by_user: dict[str, List[str]] = {}
        profile_users = await self._external.list_profile_users(tenant_id)
        for user_id in profile_users:
            facts = await self._external.fetch_profile_facts(user_id, tenant_id)
            keys_by_user[user_id] = [f.key for f in facts if f.key]

        deleted = await self._external.purge_tenant_profiles(tenant_id)

        user_keys_stripped = 0
        if self._purge_tenant_l4_strip_user_keys:
            user_ids = set(profile_users) | set(self._hot.list_user_ids(tenant_id))
            for user_id in user_ids:
                keys = keys_by_user.get(user_id, [])
                user_keys_stripped += self._hot.strip_user_keys(
                    tenant_id, user_id, keys
                )

        audit_deleted = 0
        if self._purge_delete_external_audit:
            from agent_platform.memory.adapters.compliance_utils import (
                delete_external_fact_audit_for_tenant,
            )

            audit_deleted = await delete_external_fact_audit_for_tenant(
                self._archive_db, tenant_id
            )

        return {
            "tenant_id": tenant_id,
            "profiles_deleted": deleted,
            "user_keys_stripped": user_keys_stripped,
            "external_audit_deleted": audit_deleted,
            "success": True,
        }

    async def _merge_l4_facts_into_user(
        self, context: RequestContext
    ) -> List[dict]:
        from agent_platform.memory.adapters.compliance_utils import (
            append_audit_log,
            content_sha256,
        )

        facts = await self._external.fetch_profile_facts(
            context.user_id, context.tenant_id
        )
        if not facts:
            return []
        deltas = [
            MemoryDelta(key=f.key, value=f.value, source=f.source or "external")
            for f in facts
        ]
        changes = self._hot.merge_user_facts_upsert(
            context.tenant_id, context.user_id, deltas
        )
        self._hot.invalidate_cache(context.tenant_id, context.user_id)
        for change in changes:
            await append_audit_log(
                self._archive_db,
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                resource_type="external_fact",
                resource_id=str(change["key"]),
                content_hash=content_sha256(str(change.get("new") or "")),
                action="merge",
                meta=change,
            )
        return changes

    async def purge_user_data(self, tenant_id: str, user_id: str) -> dict:
        from agent_platform.memory.adapters.rag_purge_utils import (
            purge_rag_documents_for_user,
        )

        summary: dict = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "messages_anonymized": 0,
            "cold_archives_deleted": 0,
            "rag_documents_deleted": 0,
            "rag_delete_errors": 0,
        }

        l4_keys: List[str] = []
        try:
            facts = await self._external.fetch_profile_facts(user_id, tenant_id)
            l4_keys = [f.key for f in facts if f.key]
        except Exception as e:
            self._logger.warning("Fetch L4 facts before purge failed: %s", e)

        self._hot.clear_user(tenant_id, user_id)
        if self._archive_db is not None:
            summary["messages_anonymized"] = await self._archive_db.anonymize_user_data(
                tenant_id, user_id
            )
        if self._cold_archive is not None:
            try:
                summary["cold_archives_deleted"] = (
                    await self._cold_archive.delete_cold_archives_for_user(
                        tenant_id, user_id
                    )
                )
            except Exception as e:
                self._logger.warning("Cold archive purge user failed: %s", e)
        if self._vector_index is not None:
            try:
                await self._vector_index.delete_user_messages(tenant_id, user_id)
            except Exception as e:
                self._logger.warning("Session vector purge user failed: %s", e)

        rag_result = await purge_rag_documents_for_user(
            self._index_port, self._archive_db, tenant_id, user_id
        )
        summary["rag_documents_deleted"] = rag_result.get("deleted", 0)
        summary["rag_delete_errors"] = rag_result.get("errors", 0)
        summary["rag_doc_ids"] = rag_result.get("doc_ids", [])

        if self._skill_memory is not None:
            l3 = await self._skill_memory.purge_l3_for_user_async(
                tenant_id, user_id
            )
            summary.update(l3)

        if self._external is not None:
            try:
                summary["external_profile_deleted"] = await self._external.delete_profile(
                    tenant_id, user_id
                )
            except Exception as e:
                self._logger.warning("External profile purge user failed: %s", e)
                summary["external_profile_deleted"] = False

        if l4_keys:
            summary["user_l4_keys_cleared"] = len(l4_keys)

        if self._purge_delete_external_audit:
            from agent_platform.memory.adapters.compliance_utils import (
                delete_external_fact_audit_for_user,
            )

            try:
                summary["external_audit_deleted"] = (
                    await delete_external_fact_audit_for_user(
                        self._archive_db, tenant_id, user_id
                    )
                )
            except Exception as e:
                self._logger.warning("External audit purge user failed: %s", e)
                summary["external_audit_deleted"] = 0

        ctx = RequestContext(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id="purge",
            trace_id="purge",
            channel="system",
        )
        await self._invalidate_session_search_cache(ctx, reason="purge")
        return summary

    async def backfill_cold_search_index(
        self,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict:
        if self._cold_archive is None:
            return {
                "indexed": 0,
                "skipped": 0,
                "errors": 0,
                "reason": "cold_archive_not_configured",
            }
        backfiller = getattr(self._cold_archive, "backfill_search_index", None)
        if backfiller is None:
            return {
                "indexed": 0,
                "skipped": 0,
                "errors": 0,
                "reason": "backfill_not_supported",
            }
        return await backfiller(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            limit=limit,
            force=force,
            dry_run=dry_run,
        )

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
        if (
            self._vector_index is not None
            and not self._cold_archive_keep_vectors
        ):
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
        if (
            self._vector_index is not None
            and not self._cold_archive_keep_vectors
        ):
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
        status = "healthy"
        out: dict = {
            "status": status,
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
            "cold_archive_keep_vectors": self._cold_archive_keep_vectors,
        }
        if self._cold_archive is not None:
            store = getattr(self._cold_archive, "_store", None)
            if store is not None and hasattr(store, "health"):
                try:
                    raw = store.health()
                    store_health = raw if isinstance(raw, dict) else {"status": "unknown"}
                    out["object_store"] = store_health
                    if store_health.get("status") not in ("healthy", "skipped"):
                        out["status"] = "degraded"
                except Exception as exc:
                    out["object_store"] = {"status": "unhealthy", "error": str(exc)}
                    out["status"] = "degraded"
        return out
