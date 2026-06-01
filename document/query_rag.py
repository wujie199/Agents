#!/usr/bin/env python3
"""RAG 离线查询 CLI：向量 + BM25 混合检索 → 加权融合 → Rerank。"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.domain.context import ACL, RequestContext
from document.rag.application.retrieval.hybrid_pipeline import hybrid_retrieve
from document.rag.application.retrieval.tag_filter import resolve_scenario_tags
from document.rag.bootstrap.query import create_query_stack, rebuild_bm25_from_chroma

log = logging.getLogger("query_rag")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _make_context(tenant_id: str, allowed_doc_ids: list[str] | None = None) -> RequestContext:
    acl = ACL(
        doc_ids=frozenset(allowed_doc_ids or ()),
    )
    return RequestContext(
        tenant_id=tenant_id,
        user_id="cli",
        session_id="cli",
        trace_id="cli",
        channel="cli",
        acl=acl,
    )


def _print_results(query: str, bundle) -> None:
    print(f"\n查询: {query}")
    if bundle.empty or not bundle.evidences:
        print("（无结果）")
        return
    plan = bundle.plan or {}
    print(
        f"模式={plan.get('mode')}  collection={plan.get('collection')}  "
        f"weights={plan.get('hybrid_weights')}  "
        f"rerank_min_score={plan.get('rerank_min_score')}"
    )
    if plan.get("tags"):
        print(f"标签过滤: {plan.get('tags')}  match={plan.get('tag_match')}")
    for i, ev in enumerate(bundle.evidences, 1):
        backend = ev.metadata.get("retrieval_backend", "?")
        rerank = ev.metadata.get("rerank_score")
        score_parts = [f"score={ev.score:.4f}"]
        if rerank is not None:
            score_parts.append(f"rerank={float(rerank):.4f}")
        tags = ev.metadata.get("tags")
        faq_section = ev.metadata.get("faq_section")
        faq_category = ev.metadata.get("faq_category")
        faq_number = ev.metadata.get("faq_number")
        src = ev.citation or ev.metadata.get("source_path") or ""
        print(f"\n--- [{i}] {' '.join(score_parts)} backend={backend} ---")
        if src:
            print(f"来源: {src}")
        if faq_number or faq_category or faq_section:
            parts = []
            if faq_number:
                parts.append(f"题号={faq_number}")
            if faq_category:
                parts.append(f"大类={faq_category}")
            if faq_section:
                parts.append(f"章节={faq_section}")
            print(" / ".join(parts))
        if tags:
            print(f"标签: {tags}")
        text = (ev.content or "").strip()
        preview = text[:400] + ("…" if len(text) > 400 else "")
        print(preview)


async def run_query(
    query: str,
    tenant_id: str,
    data_dir: Path,
    config_dir: str,
    top_k: int | None,
    rerank_n: int | None,
    rerank_min_score: float | None,
    allowed_doc_ids: list[str] | None = None,
    tags: list[str] | None = None,
    tag_match: str = "any",
) -> int:
    stack = create_query_stack(
        data_dir,
        config_dir=config_dir,
        tenant_id=tenant_id,
    )
    ctx = _make_context(tenant_id, allowed_doc_ids)
    bundle = await hybrid_retrieve(
        query,
        ctx,
        vector_port=stack.vector_port,
        embedding_model=stack.embedding_model,
        bm25_index=stack.bm25_index,
        rerank_model=stack.rerank_model,
        config=stack.config,
        top_k=top_k,
        rerank_n=rerank_n,
        rerank_min_score=rerank_min_score,
        tags=tags,
        tag_match=tag_match,
    )
    _print_results(query, bundle)
    return 0


async def run_repl(
    tenant_id: str,
    data_dir: Path,
    config_dir: str,
    top_k: int | None,
    rerank_n: int | None,
    rerank_min_score: float | None,
    allowed_doc_ids: list[str] | None = None,
    tags: list[str] | None = None,
    tag_match: str = "any",
) -> int:
    stack = create_query_stack(
        data_dir,
        config_dir=config_dir,
        tenant_id=tenant_id,
    )
    ctx = _make_context(tenant_id, allowed_doc_ids)
    print("RAG 查询 REPL（空行退出）")
    print(f"  data_dir={data_dir}")
    print(f"  chroma={stack.chroma_dir}")
    print(f"  collection={stack.config.collection_name}")
    print(f"  hybrid={stack.config.retrieval.enable_hybrid}")
    if tags:
        print(f"  tags={tags}  tag_match={tag_match}")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break
        bundle = await hybrid_retrieve(
            query,
            ctx,
            vector_port=stack.vector_port,
            embedding_model=stack.embedding_model,
            bm25_index=stack.bm25_index,
            rerank_model=stack.rerank_model,
            config=stack.config,
            top_k=top_k,
            rerank_n=rerank_n,
            rerank_min_score=rerank_min_score,
            tags=tags,
            tag_match=tag_match,
        )
        _print_results(query, bundle)
    return 0


def _resolve_query_tags(
    scenario: str | None,
    explicit_tags: list[str] | None,
    tag_match: str | None,
    config_dir: str,
) -> tuple[list[str], str]:
    try:
        return resolve_scenario_tags(
            scenario,
            config_dir=config_dir,
            explicit_tags=explicit_tags,
            tag_match=tag_match,
        )
    except KeyError as exc:
        log.error("%s", exc)
        sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 混合检索查询 CLI")
    parser.add_argument("query", nargs="?", default=None, help="查询文本；省略则进入 REPL")
    parser.add_argument("--tenant", default="default")
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data" / "rag_offline"),
    )
    parser.add_argument(
        "--config-dir",
        default=str(REPO_ROOT / "config"),
    )
    parser.add_argument("--top-k", type=int, default=None, help="向量/BM25 各自 top_k 覆盖")
    parser.add_argument("--rerank-n", type=int, default=None)
    parser.add_argument(
        "--rerank-min-score",
        type=float,
        default=None,
        help="Rerank 分数阈值（仅保留 > 该值的结果；默认读 config/rag_pipeline.yml）",
    )
    parser.add_argument(
        "--allow-doc-id",
        action="append",
        default=None,
        dest="allow_doc_ids",
        help="ACL 允许访问的 doc_id（可重复指定；默认不限制）",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="场景名（见 config/scenarios.yml，解析为标签过滤）",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=None,
        dest="query_tags",
        help="检索标签过滤（可重复；与 --scenario 合并）",
    )
    parser.add_argument(
        "--tag-match",
        choices=("any", "all"),
        default=None,
        help="标签匹配模式：any=任一命中，all=全部命中（默认 any 或场景配置）",
    )
    parser.add_argument(
        "--rebuild-bm25",
        action="store_true",
        help="从 Chroma 全量重建 BM25 索引后退出",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.rebuild_bm25:
        n, path = rebuild_bm25_from_chroma(
            data_dir,
            config_dir=args.config_dir,
            tenant_id=args.tenant,
        )
        print(f"BM25 重建完成: {n} chunks → {path}")
        sys.exit(0)

    tags, tag_match = _resolve_query_tags(
        args.scenario,
        args.query_tags,
        args.tag_match,
        args.config_dir,
    )

    if args.query:
        sys.exit(
            asyncio.run(
                run_query(
                    args.query,
                    args.tenant,
                    data_dir,
                    args.config_dir,
                    args.top_k,
                    args.rerank_n,
                    args.rerank_min_score,
                    args.allow_doc_ids,
                    tags or None,
                    tag_match,
                )
            )
        )

    sys.exit(
        asyncio.run(
            run_repl(
                args.tenant,
                data_dir,
                args.config_dir,
                args.top_k,
                args.rerank_n,
                args.rerank_min_score,
                args.allow_doc_ids,
                tags or None,
                tag_match,
            )
        )
    )


if __name__ == "__main__":
    main()
