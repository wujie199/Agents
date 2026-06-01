"""RAG 端到端演示：KnowledgeBasePort 摄取+索引 → 检索。"""
import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.domain.context import RequestContext
from core.composition.production_factory import build_development_context
from core.ports.index import IndexProfile


async def run_e2e(file_path: str, tenant_id: str, query: str, doc_id: str) -> None:
    request = RequestContext(
        tenant_id=tenant_id,
        user_id="e2e_user",
        session_id="e2e_session",
        trace_id="e2e_trace",
        channel="cli",
    )
    ctx = build_development_context(request, data_dir=str(ROOT / "data" / "rag_e2e"))

    kb = ctx.require_knowledge_base()
    result = await kb.ingest_and_index(
        file_path=file_path,
        doc_id=doc_id,
        tenant_id=tenant_id,
        index_profile=IndexProfile.VECTOR_ONLY,
    )
    print(
        f"[ingest+index] success={result.success} "
        f"status={result.ingest.status.value if result.ingest else 'n/a'} "
        f"chunks={result.index.chunk_count if result.index else 0}"
    )
    if not result.success:
        print(f"  errors={result.errors}")
        return

    bundle = await ctx.require_rag().route_and_retrieve(query, request)
    print(f"[retrieve] empty={bundle.empty} degraded={bundle.is_degraded()} count={len(bundle.evidences)}")
    if bundle.degraded_reason:
        print(f"  reason={bundle.degraded_reason.value} code={bundle.error_code}")
    for i, ev in enumerate(bundle.evidences[:3], 1):
        preview = (ev.content or "")[:200].replace("\n", " ")
        print(f"  [{i}] score={ev.score:.4f} id={ev.id} preview={preview!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG ingest → index → retrieve demo")
    parser.add_argument("file", help="Source document path")
    parser.add_argument("--tenant", default="tenant_e2e")
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--query", default="文档主要内容是什么？")
    args = parser.parse_args()

    doc_id = args.doc_id or Path(args.file).stem
    asyncio.run(run_e2e(args.file, args.tenant, args.query, doc_id))


if __name__ == "__main__":
    main()
