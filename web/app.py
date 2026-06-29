#!/usr/bin/env python3
"""Gradio 前端 — 对话 + 知识库上传 + 用户切换。

启动:
  cd /path/to/Agents
  pip install gradio
  python web/app.py

访问:
  http://localhost:7860
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from typing import AsyncGenerator, Optional

# ── 确保项目根在 sys.path ──
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gradio as gr

# ── 后端核心模块直调 ──
from core.domain.context import RequestContext
from app.agents.context_factory import build_chat_run_context
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
_chat_cfg = None


import time

async def _get_or_create_session(
    tenant_id: str, user_id: str, session_id: str
) -> ChatSessionHandle:
    """获取或创建 ChatSessionHandle（复用 chat_server.py 的 ChatSessionRegistry 逻辑）。"""
    global _chat_cfg
    key = f"{tenant_id}:{user_id}:{session_id}"
    if key in _sessions:
        return _sessions[key]

    t0 = time.perf_counter()
    if _chat_cfg is None:
        _chat_cfg = load_chat_config(CONFIG_DIR, profile=PROFILE)
    print(f"[perf] load_chat_config: {(time.perf_counter()-t0)*1000:.0f}ms")

    t1 = time.perf_counter()
    request = RequestContext(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        trace_id="gradio",
        channel="web",
    )
    run_ctx = build_chat_run_context(
        request, profile=PROFILE, config_dir=CONFIG_DIR, data_dir=DATA_DIR
    )
    print(f"[perf] build_chat_run_context: {(time.perf_counter()-t1)*1000:.0f}ms")

    t2 = time.perf_counter()
    await bootstrap_memory_runtime(
        run_ctx, data_dir=DATA_DIR, config_dir=CONFIG_DIR, profile=PROFILE
    )
    print(f"[perf] bootstrap_memory_runtime: {(time.perf_counter()-t2)*1000:.0f}ms")

    t3 = time.perf_counter()
    handle = ChatSessionHandle(run_ctx=run_ctx, chat_cfg=_chat_cfg)
    handle.lg_session = await create_chat_langgraph_session_async(
        run_ctx, chat_cfg=_chat_cfg
    )
    print(f"[perf] create_langgraph_session: {(time.perf_counter()-t3)*1000:.0f}ms")
    print(f"[perf] TOTAL: {(time.perf_counter()-t0)*1000:.0f}ms")

    _sessions[key] = handle
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

        while True:
            try:
                payload = await asyncio.wait_for(
                    _ev_gen.__anext__(), timeout=_LLM_EVENT_TIMEOUT
                )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
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
                # RAG 检索完成，通知前端
                yield json.dumps({"type": "status", "text": "📚 RAG 检索完成，正在推理..."}, ensure_ascii=False), ""
            elif etype == "thinking":
                if t_first_thinking is None:
                    t_first_thinking = time.perf_counter()
                thinking_chunks += 1
                thinking_acc += event.get("text", "")
                # 实时更新：thinking 折叠块 + 已有 content
                thinking_html = (
                    f"<details open><summary>💭 思考过程</summary>\n\n"
                    f"{thinking_acc}\n</details>\n\n"
                )
                yield thinking_html + accumulated, _format_retrieval_markdown(
                    evidences_summary, memory_summary
                )
            elif etype == "delta":
                if t_first_delta is None:
                    t_first_delta = time.perf_counter()
                token_count += 1
                accumulated += event.get("text", "")
                # 如果有 thinking，在前面加折叠块
                if thinking_acc:
                    thinking_html = (
                        f"<details><summary>💭 思考过程（{len(thinking_acc)} 字）</summary>\n\n"
                        f"{thinking_acc}\n</details>\n\n"
                    )
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
                    thinking_html = (
                        f"<details><summary>💭 思考过程（{len(thinking_acc)} 字）</summary>\n\n"
                        f"{thinking_acc}\n</details>\n\n"
                    )
                    final_display = thinking_html + final
                else:
                    final_display = final
                if len(final_display) > len(
                    (f"<details><summary>💭 思考过程（{len(thinking_acc)} 字）</summary>\n\n"
                     f"{thinking_acc}\n</details>\n\n" if thinking_acc else "")
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
    files: list, tenant_id: str
) -> str:
    """上传文件 → 摄取 → 切块 → 向量入库。"""
    # ── 懒加载 RAG 模块，避免启动时拖慢 ──
    from core.ports.index import IndexProfile
    from document.rag.config import load_rag_pipeline_config
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
    from document.build_rag_index import (
        step1_load_config,
        step2_ingest_file,
        step3_clean_text,
        step4_tag_metadata,
        step5_chunk_embed_write,
    )

    if not files:
        return "请选择文件"

    data_dir = Path(DEFAULT_DATA_DIR)
    cfg = step1_load_config(CONFIG_DIR)
    ingest_port = build_offline_ingest_port(cfg)
    index_service, chroma_dir = create_offline_index_service(
        data_dir, cfg, config_dir=CONFIG_DIR, index_profile=IndexProfile.VECTOR_ONLY
    )
    manifest = IndexManifest.for_data_dir(data_dir)

    results = []
    for f in files:
        file_path = Path(f.name if hasattr(f, "name") else f)
        if not file_path.exists():
            results.append(f"❌ {file_path.name}: 文件不存在")
            continue
        try:
            fid = doc_id_from_file_md5(file_md5_hex(file_path))
            report = await _build_one(
                file_path, fid, tenant_id, data_dir, cfg,
                ingest_port, index_service, chroma_dir, manifest,
            )
            if report.success:
                if report.skipped:
                    results.append(f"⏭️ {file_path.name}: 已索引，跳过")
                else:
                    chunks = report.index.chunk_count if report.index else 0
                    vectors = report.index.vectors_written if report.index else 0
                    results.append(f"✅ {file_path.name}: {chunks} 切块, {vectors} 向量入库")
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

                session_status = gr.Textbox(label="会话状态", value="就绪", interactive=False)

                gr.Markdown("---")
                gr.Markdown("### 📚 知识库管理")
                file_upload = gr.File(
                    label="上传文档",
                    file_types=[".pdf", ".txt", ".md", ".docx", ".html"],
                    file_count="multiple",
                )
                upload_btn = gr.Button("📤 上传入库", variant="primary")
                upload_result = gr.Textbox(label="上传结果", interactive=False, lines=5)

                list_docs_btn = gr.Button("📋 查看已索引文档")
                docs_list = gr.Markdown("")

            with gr.Column(scale=3):
                # ── 对话区 ──
                load_older_btn = gr.Button("⬆️ 加载更早的对话", visible=False, variant="secondary")
                chatbot = gr.Chatbot(
                    label="对话",
                    height=520,
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
            # 防重入：同一 session 不允许并发处理
            handle = await _get_or_create_session(tid, uid, sid)
            if handle.lock.locked():
                yield history, "", ""
                return
            async with handle.lock:
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
        async def on_upload(files, tid):
            if files is None:
                return "请选择文件"
            return await upload_and_index(files, tid)

        upload_btn.click(
            on_upload,
            inputs=[file_upload, tenant_id],
            outputs=[upload_result],
        )

        # ── 查看已索引文档 ──
        list_docs_btn.click(
            list_indexed_docs,
            inputs=[tenant_id],
            outputs=[docs_list],
        )

        # ── 页面加载时预热会话 + 加载历史对话（初始仅最近 10 轮） ──
        _INITIAL_TURN_LIMIT = 10  # 初始显示轮数

        async def load_session_history(tid, uid, sid):
            """页面加载 / 切换 session_id 时：预热会话并加载最近 N 轮历史消息。
            
            返回:
              chatbot_history: 最近 N 轮消息（Gradio Chatbot 格式）
              all_turns_cache: 全部历史消息缓存
              visible_start_idx: 当前显示的起始索引（在缓存中的位置）
              status_text: 状态文本
              load_more_visible: "加载更多"按钮是否可见
            """
            try:
                handle = await _get_or_create_session(tid, uid, sid)
                memory = handle.run_ctx.require_memory()
                request = handle.run_ctx.request
                # 读取该 session 的所有消息
                rows = await memory.list_turns(request, limit=200)
                # 转换为 Gradio Chatbot 格式
                all_history = []
                for row in rows:
                    role = row.get("role")
                    content = row.get("content", "")
                    if role in ("user", "assistant") and content:
                        all_history.append({"role": role, "content": content})
                
                total = len(all_history)
                if total == 0:
                    return [], [], 0, "会话已就绪（无历史消息）", False
                
                # 初始只显示最近 _INITIAL_TURN_LIMIT 轮
                start_idx = max(0, total - _INITIAL_TURN_LIMIT)
                visible_history = all_history[start_idx:]
                has_more = start_idx > 0
                status = f"已加载 {len(visible_history)} 条历史消息" + ("（还有更早的对话）" if has_more else "")
                return visible_history, all_history, start_idx, status, has_more
            except Exception as exc:
                return [], [], 0, f"加载失败: {exc}", False

        async def load_older_turns(current_history, all_cache, start_idx):
            """点击"加载更早对话"按钮时：从缓存中再取前 10 轮，prepend 到 Chatbot。"""
            if not all_cache or start_idx <= 0:
                return current_history, all_cache, start_idx, gr.update(visible=False)
            
            # 向前扩展 10 轮
            new_start = max(0, start_idx - _INITIAL_TURN_LIMIT)
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

def main():
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
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
