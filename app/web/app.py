#!/usr/bin/env python3
"""Gradio 前端 — 对话 + 知识库上传 + 用户切换。

启动:
  cd /path/to/Agents
  pip install gradio
  python app/web/app.py
  python app/web/app.py --port 7861
  python app/web/app.py --debug          # 详细日志（thinking / httpx 等）
  GRADIO_SERVER_PORT=7861 python app/web/app.py

访问:
  http://localhost:7860（默认；若被占用会自动尝试下一端口）
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import uuid
import logging
from dataclasses import replace
from pathlib import Path
from typing import AsyncGenerator, Optional

_think_dbg = logging.getLogger("thinking_debug")

# ── 确保项目根在 sys.path ──
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── 页面加载性能埋点（写入 .cursor/debug-b68651.log） ──
_PERF_LOG_PATH = REPO_ROOT / ".cursor" / "debug-b68651.log"


def _web_perf_log(event: str, timings: dict, **extra) -> None:
    import time

    entry = {
        "sessionId": "b68651",
        "runId": "page-load",
        "location": "app.web",
        "message": event,
        "data": {**timings, **extra},
        "timestamp": int(time.time() * 1000),
    }
    try:
        _PERF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_PERF_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


import gradio as gr

# ── 后端核心模块直调 ──
from core.domain.context import RequestContext
from app.agents.context_factory import (
    dev_stack_is_initialized,
    get_or_build_shared_dev_context,
)
from app.agents.memory.memory_bootstrap import bootstrap_memory_runtime
from app.agents.orchestration.chat_config import load_chat_config
from app.agents.orchestration.chat_service import (
    ChatSessionHandle,
    execute_chat_turn,
    stream_chat_turn_events,
)
from app.agents.orchestration.chat_langgraph import create_chat_langgraph_session_async
from app.agents.roles.react_loop import end_agent_session

# ── RAG 建库（懒加载，仅在上传文件时才 import） ──


# ── 全局配置 ──
CONFIG_DIR = str(REPO_ROOT / "config")
DATA_DIR = str(REPO_ROOT / "data")
DEFAULT_DATA_DIR = str(REPO_ROOT / "data" / "rag_offline")
PROFILE = "dev"
ENGINE = "langgraph"


# ══════════════════════════════════════════════════════════
# RAG 检索结果格式化
# ══════════════════════════════════════════════════════════

def _format_perf_waterfall(perf: dict) -> str:
    """将 perf 事件格式化为 Markdown waterfall 图。"""
    total = perf.get("total_ms") or 0
    if total <= 0:
        return ""

    max_bar = 40  # 最大条形长度

    def bar(ms: float | None) -> str:
        if ms is None:
            return "─"
        ratio = min(ms / max(total, 1), 1.0)
        return "█" * max(1, int(ratio * max_bar))

    def fmt(ms: float | None) -> str:
        if ms is None:
            return "  N/A"
        if ms < 1000:
            return f"{ms:6.0f}ms"
        return f"{ms/1000:5.1f}s "

    rows = [
        ("会话获取", perf.get("session_ms")),
        ("Prepare", perf.get("prepare_ms")),
        ("首thinking", perf.get("thinking_first_ms")),
        ("TTFT", perf.get("ttft_ms")),
        ("思考持续", perf.get("thinking_duration_ms")),
        ("LLM流式", perf.get("llm_stream_ms")),
        ("流式总", perf.get("stream_total_ms")),
        ("端到端", perf.get("total_ms")),
    ]

    lines = ["\n\n---\n**⏱ 性能分析**\n"]
    lines.append("| 阶段 | 耗时 | 可视化 |")
    lines.append("|------|------|--------|")
    for label, ms in rows:
        lines.append(f"| {label} | {fmt(ms)} | {bar(ms)} |")

    # 摘要指标
    extras = []
    tc = perf.get("token_count")
    if tc:
        extras.append(f"{tc} tokens")
    tps = perf.get("tokens_per_sec")
    if tps:
        extras.append(f"{tps} tok/s")
    tchars = perf.get("thinking_chars")
    if tchars:
        extras.append(f"思考 {tchars} 字")
    cchars = perf.get("content_chars")
    if cchars:
        extras.append(f"输出 {cchars} 字")
    if extras:
        lines.append(f"\n📊 {' · '.join(extras)}")

    return "\n".join(lines)


def _format_rag_markdown(evidences: list[dict]) -> str:
    """将 evidences_summary 列表格式化为 Markdown 展示文本。"""
    if not evidences:
        return ""
    lines = ["#### 📚 RAG 知识库\n"]
    for i, ev in enumerate(evidences, 1):
        score = ev.get("score", 0)
        citation = ev.get("citation", "")
        preview = ev.get("content_preview", "")
        score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        lines.append(f"**{i}.** [{score_bar}] `{score:.2f}`")
        if citation:
            lines.append(f"　来源: {citation}")
        if preview:
            lines.append(f"　> {preview}")
        lines.append("")
    return "\n".join(lines)


def _format_memory_markdown(memory_summary: dict) -> str:
    """将记忆组件命中摘要格式化为 Markdown。"""
    if not memory_summary:
        return ""
    recall_hit = memory_summary.get("recall_hit")
    skill_hit = memory_summary.get("skill_hit")
    l4_hit = memory_summary.get("l4_hit")
    lines = ["#### 🧠 记忆组件\n"]
    if recall_hit:
        lines.append("- ✅ 会话回忆")
        preview = memory_summary.get("recall_preview", "")
        if preview:
            lines.append(f"  > {preview[:200]}")
    else:
        lines.append("- ⬜ 会话回忆")
    if skill_hit:
        lines.append("- ✅ 技能检索")
        preview = memory_summary.get("skill_preview", "")
        if preview:
            lines.append(f"  > {preview[:200]}")
    else:
        lines.append("- ⬜ 技能检索")
    if l4_hit:
        lines.append("- ✅ 用户画像")
        preview = memory_summary.get("l4_preview", "")
        if preview:
            lines.append(f"  > {preview[:200]}")
    else:
        lines.append("- ⬜ 用户画像")
    lines.append("")
    return "\n".join(lines)


def _format_retrieval_markdown(
    evidences: list[dict], memory_summary: dict | None = None,
) -> str:
    """合并 RAG + 记忆组件的检索结果为统一 Markdown。"""
    rag_md = _format_rag_markdown(evidences)
    mem_md = _format_memory_markdown(memory_summary or {})
    if not rag_md and not mem_md:
        return ""
    parts = ["### 🔍 检索结果\n"]
    if rag_md:
        parts.append(rag_md)
    if mem_md:
        parts.append(mem_md)
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════
# 会话管理：每个 (tenant, user, session) 对应一个 ChatSessionHandle
# ══════════════════════════════════════════════════════════

_sessions: dict[str, ChatSessionHandle] = {}
_session_chat_locks: dict[str, asyncio.Lock] = {}
_chat_cfg = None
_archive_db = None
_MESSAGE_COLS = [
    "message_id",
    "session_id",
    "role",
    "content",
    "ts",
    "token_count",
    "metadata_json",
]


def _get_archive_db():
    """历史加载专用：只开 L2 SQLite，不走 LangGraph / 向量 reindex。"""
    global _archive_db
    if _archive_db is None:
        from agent_platform.memory.adapters.config_loader import load_memory_config
        from agent_platform.memory.adapters.archive_factory import build_archive_db

        mem_cfg = load_memory_config(f"{CONFIG_DIR}/memory.yml")
        _archive_db = build_archive_db(mem_cfg, data_dir=DATA_DIR)
    return _archive_db


async def _fetch_recent_session_messages(
    session_id: str, *, fetch_limit: int = 500
) -> list[dict]:
    db = _get_archive_db()
    rows = await db.select_many(
        "messages",
        _MESSAGE_COLS,
        where={"session_id": session_id},
        order_by="ts DESC",
        limit=fetch_limit,
    )
    rows.reverse()
    return rows


def _rows_to_chatbot_history(rows: list[dict]) -> list[dict]:
    history: list[dict] = []
    for row in rows:
        role = row.get("role")
        content = row.get("content", "")
        if role in ("user", "assistant") and content:
            history.append({"role": role, "content": content})
    return history


def _session_chat_lock(tenant_id: str, user_id: str, session_id: str) -> asyncio.Lock:
    key = f"{tenant_id}:{user_id}:{session_id}"
    if key not in _session_chat_locks:
        _session_chat_locks[key] = asyncio.Lock()
    return _session_chat_locks[key]


import time


async def _background_vector_reindex(handle: ChatSessionHandle) -> None:
    """首屏预热后后台补跑向量 reindex，避免阻塞 session_status 更新。"""
    from agent_platform.memory.adapters.config_loader import load_memory_config
    from app.agents.memory.memory_bootstrap import bootstrap_session_vectors

    t0 = time.perf_counter()
    try:
        mem_cfg = load_memory_config(f"{CONFIG_DIR}/memory.yml")
        memory = handle.run_ctx.require_memory()
        result = await bootstrap_session_vectors(memory, handle.run_ctx.request, mem_cfg)
        handle._vector_reindex_done = True
        _web_perf_log(
            "vector_reindex_background",
            {"elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)},
            session_id=handle.run_ctx.request.session_id,
            result=result,
        )
    except Exception as exc:
        _web_perf_log(
            "vector_reindex_background_error",
            {"elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)},
            session_id=handle.run_ctx.request.session_id,
            error=str(exc)[:200],
        )


async def _get_or_create_session(
    tenant_id: str,
    user_id: str,
    session_id: str,
    *,
    skip_vector_reindex: bool = False,
) -> ChatSessionHandle:
    """获取或创建 ChatSessionHandle（复用 chat_server.py 的 ChatSessionRegistry 逻辑）。"""
    global _chat_cfg
    key = f"{tenant_id}:{user_id}:{session_id}"
    if key in _sessions:
        return _sessions[key]

    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    t_cfg = time.perf_counter()
    if _chat_cfg is None:
        _chat_cfg = load_chat_config(CONFIG_DIR, profile=PROFILE)
    timings["load_chat_config_ms"] = round((time.perf_counter() - t_cfg) * 1000, 1)

    t_ctx = time.perf_counter()
    stack_reused = dev_stack_is_initialized()
    request = RequestContext(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        trace_id="gradio",
        channel="web",
    )
    run_ctx = get_or_build_shared_dev_context(
        request, config_dir=CONFIG_DIR, data_dir=DATA_DIR
    )
    timings["build_run_context_ms"] = round((time.perf_counter() - t_ctx) * 1000, 1)
    timings["shared_dev_stack_reused"] = stack_reused

    t_boot = time.perf_counter()
    await bootstrap_memory_runtime(
        run_ctx,
        data_dir=DATA_DIR,
        config_dir=CONFIG_DIR,
        profile=PROFILE,
        skip_vector_reindex=skip_vector_reindex,
    )
    timings["bootstrap_ms"] = round((time.perf_counter() - t_boot) * 1000, 1)

    t_lg = time.perf_counter()
    handle = ChatSessionHandle(run_ctx=run_ctx, chat_cfg=_chat_cfg)
    handle.lg_session = await create_chat_langgraph_session_async(
        run_ctx, chat_cfg=_chat_cfg
    )
    handle._vector_reindex_done = not skip_vector_reindex
    timings["langgraph_session_ms"] = round((time.perf_counter() - t_lg) * 1000, 1)
    timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    _sessions[key] = handle
    _web_perf_log(
        "session_warmup",
        timings,
        session_id=session_id,
        skip_vector_reindex=skip_vector_reindex,
    )
    return handle


async def _end_session(tenant_id: str, user_id: str, session_id: str) -> None:
    """结束会话并释放资源。"""
    key = f"{tenant_id}:{user_id}:{session_id}"
    handle = _sessions.pop(key, None)
    if handle is None:
        return
    try:
        await end_agent_session(handle.run_ctx, chat_cfg=handle.chat_cfg)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════
# 对话逻辑（流式 yield）
# ══════════════════════════════════════════════════════════

async def chat_stream(
    message: str,
    history: list,
    tenant_id: str,
    user_id: str,
    session_id: str,
    enable_rag: bool,
) -> AsyncGenerator[tuple[str, str], None]:
    """流式对话：yield (accumulated_text, rag_markdown)，Gradio 自动追加到消息气泡。

    支持事件类型:
      - meta: RAG 检索元信息
      - thinking: 推理模型思考过程（渲染为可折叠块）
      - delta: 正式输出 token
      - done: 完成信号
      - perf: 全链路性能计时（waterfall）
    """
    if not message.strip():
        yield "请输入内容", ""
        return

    # ── 全链路计时起点 ──
    t_total_start = time.perf_counter()
    t_session_start = time.perf_counter()
    handle = await _get_or_create_session(tenant_id, user_id, session_id)
    t_session_ms = (time.perf_counter() - t_session_start) * 1000

    # 会话就绪，通知前端
    yield json.dumps({"type": "status", "text": "🔍 会话就绪，检索知识库..."}, ensure_ascii=False), ""

    accumulated = ""
    thinking_acc = ""
    evidences_summary: list[dict] = []
    memory_summary: dict = {}

    # 计时标记
    t_stream_start = time.perf_counter()
    t_first_meta: float | None = None
    t_first_thinking: float | None = None
    t_first_delta: float | None = None
    t_last_event: float = t_stream_start
    t_done: float | None = None
    token_count = 0
    thinking_chunks = 0

    try:
        _ev_gen = stream_chat_turn_events(
            handle,
            message,
            engine=ENGINE,
            enable_rag=enable_rag,
            stream_mode="auto",
        )
        _LLM_EVENT_TIMEOUT = 120  # 秒，与 DashScope timeout 对齐
        _PULSE_INTERVAL = 3  # 秒，思考脉冲间隔
        _waiting_for_llm = False  # 等待首个 LLM 内容事件
        _pulse_dots = 0  # 脉冲动画阶段
        _idle_seconds = 0.0  # 累计等待秒数

        while True:
            try:
                # 等待 LLM 内容时用短超时，允许 yield 脉冲动画
                timeout = _PULSE_INTERVAL if _waiting_for_llm else _LLM_EVENT_TIMEOUT
                payload = await asyncio.wait_for(
                    _ev_gen.__anext__(), timeout=timeout
                )
                _idle_seconds = 0.0  # 收到事件，重置
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                if _waiting_for_llm:
                    _idle_seconds += _PULSE_INTERVAL
                    if _idle_seconds >= _LLM_EVENT_TIMEOUT:
                        yield (
                            accumulated + "\n\n⏱️ LLM 响应超时（>120s 无事件），请检查网络或重试",
                            _format_retrieval_markdown(evidences_summary, memory_summary),
                        )
                        break
                    # yield 思考脉冲动画
                    _pulse_dots = (_pulse_dots % 3) + 1
                    dots = "·" * _pulse_dots
                    yield json.dumps({"type": "status", "text": f"💭 正在推理{dots}"}, ensure_ascii=False), ""
                    continue
                yield (
                    accumulated + "\n\n⏱️ LLM 响应超时（>120s 无事件），请检查网络或重试",
                    _format_retrieval_markdown(evidences_summary, memory_summary),
                )
                break

            event = json.loads(payload)
            etype = event.get("type")
            if etype == "meta":
                t_first_meta = time.perf_counter()
                evidences_summary = event.get("evidences_summary") or []
                memory_summary = event.get("memory_summary") or {}
                # RAG 检索完成，通知前端；进入 LLM 等待状态
                _waiting_for_llm = True
                _idle_seconds = 0.0
                yield json.dumps({"type": "status", "text": "📚 RAG 检索完成，正在推理..."}, ensure_ascii=False), ""
            elif etype == "thinking":
                _waiting_for_llm = False
                if t_first_thinking is None:
                    t_first_thinking = time.perf_counter()
                    _think_dbg.debug(
                        "[THINK-6 Gradio] 收到首个 thinking 事件: text=%r",
                        event.get("text", "")[:50],
                    )
                thinking_chunks += 1
                thinking_acc += event.get("text", "")
                # 实时更新：thinking 折叠块 + 已有 content
                thinking_html = f"<thinking>\n{thinking_acc}\n</thinking>\n\n"
                yield thinking_html + accumulated, _format_retrieval_markdown(
                    evidences_summary, memory_summary
                )
            elif etype == "delta":
                _waiting_for_llm = False
                if t_first_delta is None:
                    t_first_delta = time.perf_counter()
                token_count += 1
                accumulated += event.get("text", "")
                # 如果有 thinking，在前面加折叠块
                if thinking_acc:
                    thinking_html = f"<thinking>\n{thinking_acc}\n</thinking>\n\n"
                    yield thinking_html + accumulated, _format_retrieval_markdown(
                        evidences_summary, memory_summary
                    )
                else:
                    yield accumulated, _format_retrieval_markdown(evidences_summary, memory_summary)
            elif etype == "done":
                t_done = time.perf_counter()
                # 确保最终文本完整
                final = event.get("assistant_text") or accumulated
                if thinking_acc:
                    thinking_html = f"<thinking>\n{thinking_acc}\n</thinking>\n\n"
                    final_display = thinking_html + final
                else:
                    final_display = final
                if len(final_display) > len(
                    (f"<thinking>\n{thinking_acc}\n</thinking>\n\n" if thinking_acc else "")
                    + accumulated
                ):
                    yield final_display, _format_retrieval_markdown(evidences_summary, memory_summary)
    except Exception as exc:
        t_done = time.perf_counter()
        yield accumulated + f"\n\n[错误] {exc}", _format_retrieval_markdown(evidences_summary, memory_summary)

    # ── 计算全链路 perf 指标 ──
    t_total_end = t_done or time.perf_counter()
    total_ms = (t_total_end - t_total_start) * 1000
    stream_total_ms = (t_total_end - t_stream_start) * 1000

    # 读取后端精确实测子阶段耗时
    _perf_stages: list[dict] = []
    try:
        _perf_stages = list(
            (handle.run_ctx.extra or {}).get("last_turn_perf_stages") or []
        )
    except Exception:
        pass

    # prepare 阶段（到 meta 事件）
    prepare_ms = None
    if t_first_meta is not None:
        prepare_ms = (t_first_meta - t_stream_start) * 1000

    # TTFT: 首 token 时间（优先 delta，其次 thinking）
    ttft_ms = None
    if t_first_delta is not None:
        ttft_ms = (t_first_delta - t_stream_start) * 1000
    elif t_first_thinking is not None:
        ttft_ms = (t_first_thinking - t_stream_start) * 1000

    # thinking 持续时间
    thinking_duration_ms = None
    if t_first_thinking is not None and t_first_delta is not None:
        thinking_duration_ms = (t_first_delta - t_first_thinking) * 1000

    # LLM 流式输出时间
    llm_stream_ms = None
    if t_first_delta is not None and t_done is not None:
        llm_stream_ms = (t_done - t_first_delta) * 1000

    # tokens/s
    tokens_per_sec = None
    if t_first_delta is not None and t_done is not None and token_count > 0:
        llm_dur = t_done - t_first_delta
        if llm_dur > 0:
            tokens_per_sec = round(token_count / llm_dur, 1)

    perf_data = {
        "type": "perf",
        "session_ms": round(t_session_ms, 1),
        "prepare_ms": round(prepare_ms, 1) if prepare_ms is not None else None,
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
        "thinking_first_ms": round((t_first_thinking - t_stream_start) * 1000, 1) if t_first_thinking is not None else None,
        "thinking_duration_ms": round(thinking_duration_ms, 1) if thinking_duration_ms is not None else None,
        "llm_stream_ms": round(llm_stream_ms, 1) if llm_stream_ms is not None else None,
        "stream_total_ms": round(stream_total_ms, 1),
        "total_ms": round(total_ms, 1),
        "token_count": token_count,
        "thinking_chars": len(thinking_acc),
        "thinking_chunks": thinking_chunks,
        "content_chars": len(accumulated),
        "tokens_per_sec": tokens_per_sec,
        "stages": [
            {
                "stage": s.get("stage"),
                "duration_ms": s.get("duration_ms"),
                **{k: v for k, v in s.items() if k not in ("stage", "duration_ms")}
            }
            for s in _perf_stages
        ],
    }
    yield json.dumps(perf_data, ensure_ascii=False), _format_retrieval_markdown(evidences_summary, memory_summary)


# ══════════════════════════════════════════════════════════
# 知识库上传逻辑
# ══════════════════════════════════════════════════════════

async def upload_and_index(
    files: list, tenant_id: str, *, force_reindex: bool = False
) -> str:
    """上传文件 → 摄取 → 切块 → 向量入库（按扩展名自动选 faq/contract profile）。"""
    # ── 懒加载 RAG 模块，避免启动时拖慢 ──
    from core.ports.index import IndexProfile
    from document.rag.config import detect_rag_profile_for_path, resolve_rag_pipeline_config_path
    from document.rag.bootstrap.offline import (
        build_offline_ingest_port,
        create_offline_index_service,
        load_offline_config,
    )
    from document.rag.application.indexing.index_manifest import (
        IndexManifest,
        doc_id_from_file_md5,
        file_md5_hex,
    )
    from document.build_rag_index import build_one_document

    if not files:
        return "请选择文件"

    data_dir = Path(DEFAULT_DATA_DIR)
    manifest = IndexManifest.for_data_dir(data_dir)

    profile_cache: dict = {}
    results = []
    for f in files:
        file_path = Path(f.name if hasattr(f, "name") else f)
        if not file_path.exists():
            results.append(f"❌ {file_path.name}: 文件不存在")
            continue
        try:
            profile = detect_rag_profile_for_path(file_path)
            if profile not in profile_cache:
                config_path = resolve_rag_pipeline_config_path(CONFIG_DIR, profile=profile)
                cfg = load_offline_config(CONFIG_DIR, config_path=config_path)
                ingest_port = build_offline_ingest_port(cfg)
                index_service, chroma_dir = create_offline_index_service(
                    data_dir, cfg, config_dir=CONFIG_DIR, index_profile=IndexProfile.VECTOR_ONLY
                )
                profile_cache[profile] = (cfg, ingest_port, index_service, chroma_dir, config_path)

            cfg, ingest_port, index_service, chroma_dir, config_path = profile_cache[profile]
            fid = doc_id_from_file_md5(file_md5_hex(file_path))
            report = await build_one_document(
                file_path,
                fid,
                tenant_id,
                data_dir,
                CONFIG_DIR,
                IndexProfile.VECTOR_ONLY,
                cfg=cfg,
                ingest_port=ingest_port,
                index_service=index_service,
                chroma_dir=chroma_dir,
                manifest=manifest,
                skip_indexed=True,
                force_reindex=force_reindex,
                config_path=config_path,
            )
            if report.success:
                if report.skipped:
                    results.append(f"⏭️ {file_path.name}: 已索引，跳过 (profile={profile})")
                else:
                    chunks = report.index.chunk_count if report.index else 0
                    vectors = report.index.vectors_written if report.index else 0
                    results.append(
                        f"✅ {file_path.name}: {chunks} 切块, {vectors} 向量入库 (profile={profile})"
                    )
            else:
                results.append(f"❌ {file_path.name}: {', '.join(report.errors)}")
        except Exception as exc:
            results.append(f"❌ {file_path.name}: {exc}")

    return "\n".join(results)


async def _build_one(
    file_path: Path,
    doc_id: str,
    tenant_id: str,
    data_dir: Path,
    cfg,
    ingest_port,
    index_service,
    chroma_dir: Optional[str],
    manifest: IndexManifest,
):
    """单文件建库（复用 build_rag_index.build_one_document 的六步流程）。"""
    from core.ports.index import IndexProfile
    from document.build_rag_index import build_one_document

    return await build_one_document(
        file_path,
        doc_id,
        tenant_id,
        data_dir,
        CONFIG_DIR,
        IndexProfile.VECTOR_ONLY,
        cfg=cfg,
        ingest_port=ingest_port,
        index_service=index_service,
        chroma_dir=chroma_dir,
        manifest=manifest,
        skip_indexed=True,
        force_reindex=False,
    )


def list_indexed_docs(tenant_id: str) -> str:
    """列出已索引文档。"""
    from document.rag.application.indexing.index_manifest import IndexManifest

    data_dir = Path(DEFAULT_DATA_DIR)
    manifest = IndexManifest.for_data_dir(data_dir)
    data = manifest._load()

    tenants = data.get("tenants", {})
    entries = tenants.get(tenant_id, {})
    if not entries:
        return f"租户 '{tenant_id}' 暂无已索引文档"

    lines = [f"**租户 {tenant_id} 的已索引文档 ({len(entries)} 个)：**\n"]
    for md5, entry in entries.items():
        doc_id = entry.get("doc_id", "?")
        source = entry.get("source_path", "?")
        chunks = entry.get("chunk_count", 0)
        vectors = entry.get("vectors_written", 0)
        ts = entry.get("indexed_at", "?")
        name = Path(source).name if source != "?" else "?"
        lines.append(f"- **{name}**  doc_id=`{doc_id}`  切块={chunks}  向量={vectors}  时间={ts}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
# Gradio UI 构建
# ══════════════════════════════════════════════════════════

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Agents 对话平台",
    ) as demo:

        # ── 顶栏：用户信息 ──
        gr.Markdown("# 🤖 Agents 对话平台")

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                # ── 登录区 ──
                gr.Markdown("### 👤 用户设置")
                tenant_id = gr.Textbox(
                    label="租户 ID", value="tenant1", interactive=True
                )
                user_id = gr.Textbox(
                    label="用户 ID", value="user1", interactive=True
                )
                session_id = gr.Textbox(
                    label="会话 ID", value="chat1", interactive=True
                )
                enable_rag = gr.Checkbox(label="启用 RAG 知识库检索", value=True)

                new_session_btn = gr.Button("🔄 新建会话", variant="secondary")
                end_session_btn = gr.Button("⏹️ 结束当前会话", variant="stop")

                session_status = gr.Textbox(
                    label="会话状态", value="⏳ 等待加载…", interactive=False
                )

                gr.Markdown("---")
                gr.Markdown("### 📚 知识库管理")
                file_upload = gr.File(
                    label="上传文档",
                    file_types=[".pdf", ".txt", ".md", ".docx", ".html"],
                    file_count="multiple",
                )
                upload_btn = gr.Button("📤 上传入库", variant="primary")
                force_reindex = gr.Checkbox(
                    label="强制重建（覆盖同文件旧索引）",
                    value=False,
                )
                upload_result = gr.Textbox(label="上传结果", interactive=False, lines=5)

                list_docs_btn = gr.Button("📋 查看已索引文档")
                docs_list = gr.Markdown("")

            with gr.Column(scale=3):
                # ── 对话区 ──
                load_older_btn = gr.Button("⬆️ 加载更早的对话", visible=False, variant="secondary")
                chatbot = gr.Chatbot(
                    label="对话",
                    height=520,
                    allow_tags=["thinking"],
                    reasoning_tags=[("<thinking>", "</thinking>")],
                )
                chat_input = gr.Textbox(
                    label="输入消息",
                    placeholder="输入消息后按 Enter 发送...",
                    lines=3,
                    show_label=False,
                )
                send_btn = gr.Button("发送", variant="primary")
                rag_result = gr.Markdown(
                    value="",
                    label="检索结果",
                    visible=True,
                )

        # ── 状态：历史消息缓存 + 分页索引 ──
        all_turns_cache = gr.State([])
        visible_start_idx = gr.State(0)

        # ════════════════════════════════════════════════════════
        # 事件绑定
        # ════════════════════════════════════════════════════════

        # ── 流式对话 ──
        async def on_chat(message, history, tid, uid, sid, rag):
            if not message.strip():
                yield history, "", ""
                return
            chat_lock = _session_chat_lock(tid, uid, sid)
            if chat_lock.locked():
                _web_perf_log(
                    "chat_skipped_duplicate",
                    {},
                    session_id=sid,
                    reason="lock_busy",
                    message_preview=message.strip()[:40],
                )
                yield history, "", ""
                return
            async with chat_lock:
                handle = await _get_or_create_session(tid, uid, sid)
                # Gradio 6.x Chatbot 不会自动追加用户消息，需手动追加
                history = history + [{"role": "user", "content": message}]
                # 流式追加 assistant 消息
                partial_history = list(history)
                partial_history.append({"role": "assistant", "content": "⏳ 连接中..."})
                yield partial_history, "", ""
                first_chunk = True
                async for text, rag_md in chat_stream(message, history, tid, uid, sid, rag):
                    # 处理 status 事件：更新占位状态文本
                    if text.startswith('{"type":"status"') or text.startswith('{"type": "status"'):
                        try:
                            status_event = json.loads(text)
                            status_text = status_event.get("text", "")
                            if status_text:
                                partial_history[-1]["content"] = status_text
                                yield partial_history, "", ""
                        except (json.JSONDecodeError, KeyError):
                            pass
                        continue
                    # 处理 perf 事件：渲染 waterfall 并追加到检索结果（记忆组件之后）
                    if text.startswith('{"type":"perf"') or text.startswith('{"type": "perf"'):
                        try:
                            perf_event = json.loads(text)
                            perf_waterfall = _format_perf_waterfall(perf_event)
                            # 将性能面板追加到 rag_result 尾部（记忆组件之后）
                            combined_rag_md = rag_md + perf_waterfall if perf_waterfall else rag_md
                            yield partial_history, "", combined_rag_md
                        except (json.JSONDecodeError, KeyError):
                            pass
                        continue
                    partial_history[-1]["content"] = text
                    # 首个真实内容到达时清除"思考中"占位
                    if first_chunk and text and "⏳" not in text:
                        first_chunk = False
                    yield partial_history, "", rag_md

        send_btn.click(
            on_chat,
            inputs=[chat_input, chatbot, tenant_id, user_id, session_id, enable_rag],
            outputs=[chatbot, chat_input, rag_result],
        )
        chat_input.submit(
            on_chat,
            inputs=[chat_input, chatbot, tenant_id, user_id, session_id, enable_rag],
            outputs=[chatbot, chat_input, rag_result],
        )

        # ── 新建会话 ──
        def on_new_session(tid, uid):
            new_sid = f"chat-{uuid.uuid4().hex[:8]}"
            return new_sid, f"新会话已创建: {new_sid}"

        new_session_btn.click(
            on_new_session,
            inputs=[tenant_id, user_id],
            outputs=[session_id, session_status],
        )

        # ── 结束会话 ──
        async def on_end_session(tid, uid, sid):
            await _end_session(tid, uid, sid)
            return "会话已结束（L4→L1 finalize 完成）"

        end_session_btn.click(
            on_end_session,
            inputs=[tenant_id, user_id, session_id],
            outputs=[session_status],
        )

        # ── 文件上传入库 ──
        async def on_upload(files, tid, force):
            if files is None:
                return "请选择文件"
            return await upload_and_index(files, tid, force_reindex=bool(force))

        upload_btn.click(
            on_upload,
            inputs=[file_upload, tenant_id, force_reindex],
            outputs=[upload_result],
        )

        # ── 查看已索引文档 ──
        list_docs_btn.click(
            list_indexed_docs,
            inputs=[tenant_id],
            outputs=[docs_list],
        )

        # ── 页面加载时加载历史对话（轻量读 L2；初始显示最近 10 轮 ≈ 20 条） ──
        _HISTORY_FETCH_LIMIT = 500
        _INITIAL_VISIBLE_MESSAGES = 20

        async def load_session_history(tid, uid, sid):
            """页面加载：分阶段更新 session_status + 历史 + 会话预热。"""
            t_total = time.perf_counter()
            timings: dict[str, float] = {}

            # 阶段 1：立即反馈（session_status 第 4 个 output）
            yield [], [], 0, "⏳ 正在加载历史…", gr.update(visible=False)

            try:
                # 阶段 2：轻量读 L2
                t_hist = time.perf_counter()
                t_db = time.perf_counter()
                rows = await _fetch_recent_session_messages(
                    sid, fetch_limit=_HISTORY_FETCH_LIMIT
                )
                timings["archive_open_and_query_ms"] = round(
                    (time.perf_counter() - t_db) * 1000, 1
                )
                all_history = _rows_to_chatbot_history(rows)
                total = len(all_history)
                timings["history_ms"] = round((time.perf_counter() - t_hist) * 1000, 1)

                if total == 0:
                    yield [], [], 0, "⏳ 正在预热会话…", gr.update(visible=False)
                else:
                    start_idx = max(0, total - _INITIAL_VISIBLE_MESSAGES)
                    visible_history = all_history[start_idx:]
                    has_more = start_idx > 0
                    hist_status = (
                        f"✓ 历史 {len(visible_history)}/{total} 条"
                        f"（{timings['history_ms']:.0f}ms）"
                    )
                    yield (
                        visible_history,
                        all_history,
                        start_idx,
                        f"⏳ 正在预热会话… | {hist_status}",
                        has_more,
                    )

                # 阶段 3：预热对话引擎（跳过向量 reindex，后台补跑）
                key = f"{tid}:{uid}:{sid}"
                t_warm = time.perf_counter()
                if key not in _sessions:
                    handle = await _get_or_create_session(
                        tid, uid, sid, skip_vector_reindex=True
                    )
                    if not getattr(handle, "_vector_reindex_done", False):
                        asyncio.create_task(_background_vector_reindex(handle))
                else:
                    handle = _sessions[key]
                timings["warmup_ms"] = round((time.perf_counter() - t_warm) * 1000, 1)
                timings["total_ms"] = round((time.perf_counter() - t_total) * 1000, 1)

                user_msgs = [m["content"] for m in all_history if m["role"] == "user"]
                _web_perf_log(
                    "page_load_complete",
                    timings,
                    session_id=sid,
                    history_count=total,
                    last_user_preview=user_msgs[-1][:40] if user_msgs else "",
                )

                if total == 0:
                    status = (
                        f"会话已就绪（无历史）| 预热 {timings['warmup_ms']:.0f}ms"
                        f" · 合计 {timings['total_ms']:.0f}ms"
                    )
                    yield [], [], 0, status, False
                else:
                    start_idx = max(0, total - _INITIAL_VISIBLE_MESSAGES)
                    visible_history = all_history[start_idx:]
                    has_more = start_idx > 0
                    status = (
                        f"会话已就绪 | 历史 {timings['history_ms']:.0f}ms"
                        f" + 预热 {timings['warmup_ms']:.0f}ms"
                        f" = {timings['total_ms']:.0f}ms"
                        + (f" · 共 {total} 条" if has_more else "")
                    )
                    yield visible_history, all_history, start_idx, status, has_more

            except Exception as exc:
                timings["total_ms"] = round((time.perf_counter() - t_total) * 1000, 1)
                _web_perf_log(
                    "page_load_error",
                    timings,
                    session_id=sid,
                    error=str(exc)[:300],
                )
                yield [], [], 0, f"加载失败: {exc}", False

        async def load_older_turns(current_history, all_cache, start_idx):
            """点击"加载更早对话"按钮时：从缓存中再取前 20 条，prepend 到 Chatbot。"""
            if not all_cache or start_idx <= 0:
                return current_history, all_cache, start_idx, gr.update(visible=False)

            new_start = max(0, start_idx - _INITIAL_VISIBLE_MESSAGES)
            older_turns = all_cache[new_start:start_idx]
            # prepend 到当前 history 前面
            updated_history = older_turns + current_history
            has_more = new_start > 0
            return updated_history, all_cache, new_start, gr.update(visible=has_more)

        demo.load(
            load_session_history,
            inputs=[tenant_id, user_id, session_id],
            outputs=[chatbot, all_turns_cache, visible_start_idx, session_status, load_older_btn],
        )

        # ── 切换 session_id 时自动加载该会话历史 ──
        session_id.change(
            load_session_history,
            inputs=[tenant_id, user_id, session_id],
            outputs=[chatbot, all_turns_cache, visible_start_idx, session_status, load_older_btn],
        )

        # ── 加载更早对话 ──
        load_older_btn.click(
            load_older_turns,
            inputs=[chatbot, all_turns_cache, visible_start_idx],
            outputs=[chatbot, all_turns_cache, visible_start_idx, load_older_btn],
        )

    return demo


# ══════════════════════════════════════════════════════════
# 启动入口
# ══════════════════════════════════════════════════════════

def _find_free_port(start: int, attempts: int = 20) -> int:
    import socket

    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise OSError(f"无法在 {start}-{start + attempts - 1} 范围内找到可用端口")


def _resolve_server_port(cli_port: int | None) -> int:
    if cli_port is not None:
        return cli_port
    env_port = os.environ.get("GRADIO_SERVER_PORT")
    if env_port:
        return int(env_port)
    preferred = 7860
    free = _find_free_port(preferred)
    if free != preferred:
        print(
            f"端口 {preferred} 已被占用，改用 {free}。"
            f"可设置 GRADIO_SERVER_PORT 或 --port 指定端口。",
            file=sys.stderr,
        )
    return free


def _configure_logging(*, debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    _think_dbg.setLevel(logging.DEBUG if debug else logging.WARNING)
    if not debug:
        for name in (
            "asyncio",
            "httpcore",
            "httpx",
            "urllib3",
            "huggingface",
            "app.agents.middleware",
            "app.agents.middleware.timing",
            "rag.router",
            "document.rag",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agents Gradio Web UI")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="监听端口（默认 7860；被占用时自动递增）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="输出详细调试日志",
    )
    args = parser.parse_args()
    debug = args.debug or os.environ.get("WEB_DEBUG", "").lower() in (
        "1",
        "true",
        "yes",
    )

    _configure_logging(debug=debug)

    # ── 绕过系统代理访问 localhost（Clash/V2Ray 等会拦截导致 502） ──
    _local_bypass = "localhost,127.0.0.1,0.0.0.0,::1"
    os.environ["no_proxy"] = f"{_local_bypass},{os.environ.get('no_proxy', '')}"
    os.environ["NO_PROXY"] = f"{_local_bypass},{os.environ.get('NO_PROXY', '')}"

    demo = build_ui()
    server_port = _resolve_server_port(args.port)
    print(f"Web UI: http://localhost:{server_port}", file=sys.stderr)
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
        .sidebar {max-width: 280px;}
        .chat-area {min-height: 500px;}
        """,
    )


if __name__ == "__main__":
    main()
