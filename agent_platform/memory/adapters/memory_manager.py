# -*- coding: utf-8 -*-
"""Hermes L4 MemoryManager：builtin L1 + 至多一个外部 Provider。"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

from core.domain.context import RequestContext
from core.ports.memory.provider import MemoryProvider, MemorySnippet

from agent_platform.memory.adapters.memory_fence import build_memory_context_block
from agent_platform.memory.adapters.memory_provider_factory import (
    build_external_memory_provider,
)
from agent_platform.memory.adapters.memory_providers.builtin_provider import (
    BuiltinProvider,
)
from agent_platform.memory.adapters.memory_security import scan_memory_content

logger = logging.getLogger(__name__)


class _CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout_s: float = 120.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._failures = 0
        self._opened_at: Optional[float] = None

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.reset_timeout_s:
            self._failures = 0
            self._opened_at = None
            return False
        return True


class MemoryManager:
    """编排 builtin + external Provider；fail-open。"""

    def __init__(
        self,
        *,
        builtin: Optional[MemoryProvider] = None,
        external: Optional[MemoryProvider] = None,
        prefetch_timeout_ms: float = 1500.0,
        circuit_breaker: Optional[_CircuitBreaker] = None,
        enabled_toolsets: Optional[list[str]] = None,
    ):
        self._builtin = builtin or BuiltinProvider()
        self._external = external
        self._prefetch_timeout_s = max(0.001, prefetch_timeout_ms / 1000.0)
        self._circuit = circuit_breaker or _CircuitBreaker()
        self._enabled_toolsets = enabled_toolsets or []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mem-sync")
        self._pending_prefetch: Optional[Future[list[MemorySnippet]]] = None
        self._pending_session_id: Optional[str] = None
        self._sync_futures: list[Future[None]] = []
        self._logger = logger

    @property
    def builtin(self) -> MemoryProvider:
        return self._builtin

    @property
    def external(self) -> Optional[MemoryProvider]:
        return self._external

    def register_external(self, provider: MemoryProvider) -> None:
        if self._external is not None:
            raise ValueError(
                f"External provider already registered: {self._external.name}"
            )
        self._external = provider

    def _active_providers(self) -> list[MemoryProvider]:
        providers: list[MemoryProvider] = [self._builtin]
        if self._external is not None and self._external.is_available():
            if not self._circuit.is_open():
                providers.append(self._external)
        return providers

    async def initialize_all(self, context: RequestContext) -> None:
        for provider in self._active_providers():
            try:
                await provider.initialize(context)
            except Exception as exc:
                self._logger.warning(
                    "Provider %s initialize failed: %s", provider.name, exc
                )

    async def prefetch_all(
        self, user_message: str, context: RequestContext
    ) -> list[MemorySnippet]:
        """并行 prefetch，超时 fail-open。"""
        snippets: list[MemorySnippet] = list(
            await self._maybe_consume_queued_prefetch(context)
        )
        providers = self._active_providers()

        async def _prefetch_one(provider: MemoryProvider) -> list[MemorySnippet]:
            try:
                await provider.on_turn_start(user_message, context)
                return await provider.prefetch_turn(user_message, context)
            except Exception as exc:
                if provider is self._external:
                    self._circuit.record_failure()
                self._logger.warning(
                    "Provider %s prefetch failed: %s", provider.name, exc
                )
                return []

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[_prefetch_one(p) for p in providers]),
                timeout=self._prefetch_timeout_s,
            )
            for batch in results:
                snippets.extend(batch)
            if self._external is not None:
                self._circuit.record_success()
        except asyncio.TimeoutError:
            if self._external is not None:
                self._circuit.record_failure()
            self._logger.warning(
                "prefetch_all timed out after %.1fs (fail-open)",
                self._prefetch_timeout_s,
            )
        return self._sanitize_snippets(snippets)

    async def _maybe_consume_queued_prefetch(
        self, context: RequestContext
    ) -> list[MemorySnippet]:
        future = self._pending_prefetch
        if future is None:
            return []
        if self._pending_session_id != context.session_id:
            return []
        if not future.done():
            return []
        snippets: list[MemorySnippet] = []
        try:
            snippets = list(future.result() or [])
            self._logger.debug(
                "Consumed queued prefetch for session=%s count=%d",
                context.session_id,
                len(snippets),
            )
        except Exception as exc:
            self._logger.debug("Queued prefetch failed: %s", exc)
        finally:
            self._pending_prefetch = None
            self._pending_session_id = None
        return snippets

    def queue_prefetch(
        self, user_message: str, context: RequestContext
    ) -> None:
        """后台预取下一回合（best-effort）。"""
        providers = self._active_providers()

        def _run() -> list[MemorySnippet]:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)

                async def _inner() -> list[MemorySnippet]:
                    out: list[MemorySnippet] = []
                    for provider in providers:
                        try:
                            batch = await provider.prefetch_turn(
                                user_message, context
                            )
                            out.extend(batch)
                        except Exception as exc:
                            self._logger.debug(
                                "queue_prefetch %s failed: %s",
                                provider.name,
                                exc,
                            )
                    return out

                return loop.run_until_complete(_inner())
            finally:
                loop.close()

        self._pending_prefetch = self._executor.submit(_run)
        self._pending_session_id = context.session_id

    def sync_all(
        self,
        user_message: str,
        assistant_message: str,
        messages: list[dict[str, Any]],
        context: RequestContext,
    ) -> None:
        """后台单线程 sync（不阻塞主路径）。"""

        def _run() -> None:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)

                async def _inner() -> None:
                    for provider in self._active_providers():
                        try:
                            await provider.sync_turn(
                                user_message,
                                assistant_message,
                                messages,
                                context,
                            )
                        except Exception as exc:
                            self._logger.warning(
                                "Provider %s sync_turn failed: %s",
                                provider.name,
                                exc,
                            )

                loop.run_until_complete(_inner())
            finally:
                loop.close()

        fut = self._executor.submit(_run)
        self._sync_futures.append(fut)

    def notify_memory_tool_write(
        self,
        tenant_id: str,
        user_id: str,
        action: str,
        payload: dict[str, Any],
    ) -> None:
        """L1 写入成功后镜像到 external Provider。"""
        if self._external is None or not self._external.is_available():
            return
        if self._circuit.is_open():
            return

        def _run() -> None:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    self._external.on_memory_write(  # type: ignore[union-attr]
                        tenant_id, user_id, action, payload
                    )
                )
            except Exception as exc:
                self._circuit.record_failure()
                self._logger.warning("on_memory_write mirror failed: %s", exc)
            else:
                self._circuit.record_success()
            finally:
                loop.close()

        fut = self._executor.submit(_run)
        self._sync_futures.append(fut)

    async def collect_pre_compress(
        self, messages: list[dict[str, Any]], context: RequestContext
    ) -> str:
        parts: list[str] = []
        for provider in self._active_providers():
            try:
                text = await provider.on_pre_compress(messages, context)
                if text and text.strip():
                    parts.append(f"[{provider.name}]\n{text.strip()}")
            except Exception as exc:
                self._logger.warning(
                    "Provider %s on_pre_compress failed: %s", provider.name, exc
                )
        return "\n\n".join(parts)

    def build_memory_context_block(
        self, snippets: list[MemorySnippet], tenant_id: str
    ) -> str:
        if not snippets:
            return ""
        body_parts: list[str] = []
        for snip in snippets:
            if snip.content.strip():
                body_parts.append(snip.content.strip())
        if not body_parts:
            return ""
        source = snippets[0].source if len(snippets) == 1 else "memory_manager"
        return build_memory_context_block("\n\n".join(body_parts), source, tenant_id)

    @staticmethod
    def inject_memory_context_into_messages(
        messages: list[dict[str, Any]], memory_block: str
    ) -> list[dict[str, Any]]:
        """将 memory block 注入当前回合 user 消息（瞬态，不落库）。"""
        if not memory_block or not messages:
            return messages
        result = [dict(m) for m in messages]
        for idx in range(len(result) - 1, -1, -1):
            if result[idx].get("role") == "user":
                content = str(result[idx].get("content") or "")
                result[idx]["content"] = f"{memory_block}\n\n{content}"
                break
        return result

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for provider in self._active_providers():
            if provider is self._builtin:
                continue
            schemas.extend(provider.get_tool_schemas())
        return schemas

    async def on_session_end(self, context: RequestContext) -> None:
        for provider in self._active_providers():
            try:
                await provider.on_session_end(context)
            except Exception as exc:
                self._logger.warning(
                    "Provider %s on_session_end failed: %s", provider.name, exc
                )

    def flush_pending(self, timeout: float = 5.0) -> None:
        futures = [*self._sync_futures]
        if self._pending_prefetch is not None:
            futures.append(self._pending_prefetch)
        for fut in futures:
            try:
                fut.result(timeout=timeout)
            except Exception as exc:
                self._logger.debug("flush_pending: %s", exc)
        self._sync_futures.clear()
        self._pending_prefetch = None
        self._pending_session_id = None

    def shutdown_all(self, timeout: float = 5.0) -> None:
        self.flush_pending(timeout=timeout)

        def _shutdown_providers() -> None:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)

                async def _inner() -> None:
                    for provider in (self._builtin, self._external):
                        if provider is None:
                            continue
                        try:
                            await provider.shutdown()
                        except Exception as exc:
                            self._logger.warning(
                                "Provider %s shutdown failed: %s",
                                provider.name,
                                exc,
                            )

                loop.run_until_complete(_inner())
            finally:
                loop.close()

        try:
            self._executor.submit(_shutdown_providers).result(timeout=timeout)
        except Exception as exc:
            self._logger.debug("shutdown_all providers: %s", exc)
        self._executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _sanitize_snippets(snippets: list[MemorySnippet]) -> list[MemorySnippet]:
        clean: list[MemorySnippet] = []
        for snip in snippets:
            threat = scan_memory_content(snip.content)
            if threat:
                logger.warning(
                    "Dropped prefetch snippet from %s: %s", snip.source, threat
                )
                continue
            clean.append(snip)
        return clean


def build_memory_manager(
    cfg: dict[str, Any],
    *,
    external_memory: Any = None,
) -> MemoryManager:
    """从 memory 配置构建 MemoryManager。"""
    cb_cfg = cfg.get("circuit_breaker") or {}
    if not isinstance(cb_cfg, dict):
        cb_cfg = {}
    circuit = _CircuitBreaker(
        failure_threshold=int(cb_cfg.get("failure_threshold", 5)),
        reset_timeout_s=float(cb_cfg.get("reset_timeout_s", 120)),
    )
    external = build_external_memory_provider(cfg, external_memory=external_memory)
    enabled = cfg.get("enabled_toolsets") or []
    if isinstance(enabled, str):
        enabled = [enabled]
    return MemoryManager(
        external=external,
        prefetch_timeout_ms=float(cfg.get("prefetch_timeout_ms", 1500)),
        circuit_breaker=circuit,
        enabled_toolsets=list(enabled),
    )


def build_on_memory_write_from_manager(
    manager: Optional[MemoryManager],
) -> Callable[[str, str, str, dict[str, Any]], None]:
    if manager is None:
        from agent_platform.memory.adapters.l1_memory_callbacks import (
            NoOpMemoryWriteMirror,
        )

        return NoOpMemoryWriteMirror()
    return manager.notify_memory_tool_write
