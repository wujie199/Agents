#!/usr/bin/env python3
"""
离线 RAG 建库 — 全流程在本文件可见

  ┌─────────────────────────────────────────────────────────────────┐
  │  [1] 读配置    config/rag_pipeline.yml → RagPipelineConfig     │
  │  [2] 摄取      PDF/Word/图/HTML → OCR；txt/md → 直读           │
  │  [3] 清理      postprocess_ocr + CompositeCleaner              │
  │  [4] 打标      规则/关键词 metadata（config/metadata_tagging.yml）│
  │  [5] 切块      RecursiveChunker（在 IndexService 内）           │
  │  [6] 向量入库   Embedder → Chroma (data/.../chroma_dev)         │
  │  [6b] BM25     同步写入 data/.../bm25_index/{collection}.json │
  └─────────────────────────────────────────────────────────────────┘

实现模块：document/rag/bootstrap、application/*、adapters/*、ocr/*

用法:
  conda activate py3.11
  pip install aiosqlite chromadb sentence-transformers  # 按需

  python document/build_rag_index.py
  # 默认扫描 document/rag/pdf 下 *.pdf；也可显式传 path
  # 强制重建：--force-reindex
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 无命令行 path 时使用：扫描该目录下的 PDF 并建库
DEFAULT_SCAN_PATH = REPO_ROOT / "document" / "rag" / "pdf"
DEFAULT_SCAN_GLOB = "*.pdf"

from core.ports.index import IndexProfile, IndexResult
from core.ports.ingest import IngestResult, IngestStatus
from document.rag.config import RagPipelineConfig, compute_index_config_hash, load_rag_pipeline_config
from document.rag.application.cleaning.pipeline import apply_ingest_cleaning
from document.rag.application.metadata.pipeline import apply_metadata_enrichment
from document.rag.application.retrieval.tag_filter import merge_tags_into_metadata
from document.rag.application.ingest.factory import detect_format
from document.rag.application.indexing.index_manifest import (
    IndexManifest,
    doc_id_from_file_md5,
    file_md5_hex,
)
from document.rag.bootstrap.offline import (
    build_offline_ingest_port,
    create_offline_index_service,
    load_offline_config,
)
from document.rag.bootstrap.query import rebuild_bm25_from_chroma

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("build_rag_index")


# ---------------------------------------------------------------------------
# 六步流水线（打开本文件即可阅读主流程）
# ---------------------------------------------------------------------------


def step1_load_config(config_dir: str) -> RagPipelineConfig:
    """[1/6] 加载 rag_pipeline.yml。"""
    cfg = load_offline_config(config_dir)
    log.info(
        "[1/6] config  collection=%s  ingest.mode=%s  cleaning=%s  metadata=%s  chunk=%s/%s strategy=%s",
        cfg.collection_name,
        cfg.ingest.mode,
        cfg.ingest.enable_cleaning,
        cfg.metadata.enabled,
        cfg.chunk_size,
        cfg.chunk_overlap,
        cfg.chunk_strategy,
    )
    return cfg


def step2_ingest_file(
    file_path: Path,
    doc_id: str,
    tenant_id: str,
    cfg: RagPipelineConfig,
    ingest_port,
    file_md5: Optional[str] = None,
) -> IngestResult:
    """[2/6] 文件 → 正文（OCR 或纯文本）。"""
    md5 = file_md5 or file_md5_hex(file_path)
    metadata = {
        "tenant_id": tenant_id,
        "doc_id": doc_id,
        "file_md5": md5,
        "source_path": str(file_path.resolve()),
    }
    result = ingest_port.ingest_from_path(
        str(file_path.resolve()),
        doc_id,
        metadata=metadata,
    )
    backend = result.metadata.get("ingest_backend", "?")
    log.info(
        "[2/6] ingest  file=%s  status=%s  backend=%s  chars=%d  pages=%d",
        file_path.name,
        result.status.value,
        backend,
        len(result.content or ""),
        len(result.pages),
    )
    return result


def step3_clean_text(
    ingest: IngestResult,
    file_path: Path,
    cfg: RagPipelineConfig,
) -> IngestResult:
    """[3/6] OCR 后处理 + 企业清洗链。"""
    if ingest.status == IngestStatus.FAILED:
        log.warning("[3/6] clean  skipped (ingest failed)")
        return ingest
    doc_format = detect_format(str(file_path))
    chars = apply_ingest_cleaning(ingest, doc_format, cfg)
    log.info(
        "[3/6] clean  format=%s  cleaned=%s  chars=%d",
        doc_format.value,
        ingest.metadata.get("cleaned", False),
        chars,
    )
    return ingest


def step4_tag_metadata(
    ingest: IngestResult,
    file_path: Path,
    cfg: RagPipelineConfig,
    extra_tags: Optional[List[str]] = None,
) -> IngestResult:
    """[4/6] 规则/关键词 metadata 打标（tags、categories、matched_rules）。"""
    if ingest.status == IngestStatus.FAILED:
        log.warning("[4/6] metadata  skipped (ingest failed)")
        return ingest
    if cfg.metadata.enabled:
        doc_format = detect_format(str(file_path))
        ingest = apply_metadata_enrichment(ingest, doc_format, cfg)
    elif extra_tags:
        ingest.metadata = merge_tags_into_metadata(ingest.metadata, extra_tags)
        log.info("[4/6] metadata  disabled; applied manual tags=%s", extra_tags)
        return ingest
    else:
        log.info("[4/6] metadata  disabled")
        return ingest
    if extra_tags:
        ingest.metadata = merge_tags_into_metadata(ingest.metadata, extra_tags)
    log.info(
        "[4/6] metadata  tags=%s  rules=%s",
        ingest.metadata.get("tags"),
        ingest.metadata.get("matched_rules"),
    )
    return ingest


async def step5_chunk_embed_write(
    ingest: IngestResult,
    doc_id: str,
    tenant_id: str,
    cfg: RagPipelineConfig,
    index_service,
    index_profile: IndexProfile,
) -> IndexResult:
    """[5/6][6/6] 切块 → Embedding → 写入 Chroma（IndexService 内聚 5+6）。"""
    if ingest.status == IngestStatus.FAILED:
        raise ValueError(f"ingest failed: {ingest.errors}")
    index_result = await index_service.index_from_ingest(
        ingest,
        tenant_id=tenant_id,
        doc_id=doc_id,
        profile=index_profile,
    )
    log.info(
        "[5/6] chunk  doc_id=%s  chunks=%d",
        doc_id,
        index_result.chunk_count,
    )
    log.info(
        "[6/6] index  vectors=%d  collection=%s  profile=%s",
        index_result.vectors_written,
        index_result.collection,
        index_result.profile.value,
    )
    return index_result

@dataclass
class BuildReport:
    success: bool
    file_path: Path
    doc_id: str
    skipped: bool = False
    file_md5: Optional[str] = None
    ingest: Optional[IngestResult] = None
    index: Optional[IndexResult] = None
    chroma_dir: Optional[str] = None
    errors: List[str] = field(default_factory=list)


async def build_one_document(
    file_path: Path,
    doc_id: str,
    tenant_id: str,
    data_dir: Path,
    config_dir: str,
    index_profile: IndexProfile,
    cfg: Optional[RagPipelineConfig] = None,
    ingest_port=None,
    index_service=None,
    chroma_dir: Optional[str] = None,
    manifest: Optional[IndexManifest] = None,
    skip_indexed: bool = True,
    force_reindex: bool = False,
    extra_tags: Optional[List[str]] = None,
) -> BuildReport:
    """
    单文件离线建库 — 六步顺序执行（本函数即主流程入口）。
    默认按文件 MD5 跳过已索引（见 data_dir/indexed_by_md5.json）。
    """
    if cfg is None:
        cfg = step1_load_config(config_dir)

    try:
        file_md5 = file_md5_hex(file_path)
    except OSError as exc:
        return BuildReport(
            success=False,
            file_path=file_path,
            doc_id=doc_id,
            errors=[f"无法读取文件计算 MD5: {exc}"],
        )

    manifest = manifest or IndexManifest.for_data_dir(data_dir)
    config_hash = compute_index_config_hash(cfg)
    if (
        skip_indexed
        and not force_reindex
        and manifest.matches_index_config(
            tenant_id,
            file_md5,
            model_version=cfg.model_version,
            config_hash=config_hash,
        )
    ):
        entry = manifest.get_entry(tenant_id, file_md5) or {}
        log.info(
            "=== SKIP %s (MD5 已索引 doc_id=%s indexed_at=%s) ===",
            file_path.name,
            entry.get("doc_id"),
            entry.get("indexed_at"),
        )
        return BuildReport(
            success=True,
            skipped=True,
            file_path=file_path,
            doc_id=str(entry.get("doc_id") or doc_id),
            file_md5=file_md5,
            chroma_dir=chroma_dir,
        )

    log.info("=== build %s (doc_id=%s md5=%s) ===", file_path.name, doc_id, file_md5[:12])

    # [2] 摄取
    if ingest_port is None:
        ingest_port = build_offline_ingest_port(cfg)
    ingest = step2_ingest_file(
        file_path, doc_id, tenant_id, cfg, ingest_port, file_md5=file_md5
    )
    if ingest.status == IngestStatus.FAILED:
        return BuildReport(
            success=False,
            file_path=file_path,
            doc_id=doc_id,
            ingest=ingest,
            chroma_dir=chroma_dir,
            errors=list(ingest.errors) or ["ingest failed"],
        )

    # [3] 清理
    ingest = step3_clean_text(ingest, file_path, cfg)

    # [4] 元数据打标
    ingest = step4_tag_metadata(ingest, file_path, cfg, extra_tags=extra_tags)

    # [5][6] 切块 + 向量入库
    if index_service is None:
        index_service, chroma_dir = create_offline_index_service(
            data_dir,
            cfg,
            config_dir=config_dir,
            index_profile=index_profile,
        )
    if force_reindex:
        deleted = await index_service.delete_document(doc_id, tenant_id)
        if deleted:
            log.info("[5/6] force-reindex 已删除旧索引 doc_id=%s", doc_id)
    try:
        index_result = await step5_chunk_embed_write(
            ingest, doc_id, tenant_id, cfg, index_service, index_profile
        )
    except Exception as exc:
        log.error("[5/6] index failed: %s", exc)
        return BuildReport(
            success=False,
            file_path=file_path,
            doc_id=doc_id,
            ingest=ingest,
            chroma_dir=chroma_dir,
            errors=[str(exc)],
        )

    manifest.register(
        tenant_id,
        file_md5,
        doc_id=doc_id,
        source_path=str(file_path.resolve()),
        model_version=cfg.model_version,
        config_hash=config_hash,
        chunk_count=index_result.chunk_count,
        vectors_written=index_result.vectors_written,
    )
    log.info("=== OK %s (manifest updated) ===", file_path.name)
    return BuildReport(
        success=True,
        file_path=file_path,
        doc_id=doc_id,
        file_md5=file_md5,
        ingest=ingest,
        index=index_result,
        chroma_dir=chroma_dir,
    )


# ---------------------------------------------------------------------------
# CLI：批量调度
# ---------------------------------------------------------------------------


def _collect_files(target: Path, glob_pattern: str) -> List[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"路径不存在: {target}")
    return sorted(target.glob(glob_pattern))


def _parse_index_profile(name: str) -> IndexProfile:
    return {
        "vector_only": IndexProfile.VECTOR_ONLY,
        "sql_sidecar": IndexProfile.SQL_SIDECAR,
        "graph_sidecar": IndexProfile.GRAPH_SIDECAR,
        "full": IndexProfile.FULL,
    }.get(name.lower(), IndexProfile.VECTOR_ONLY)


async def run_build(
    paths: List[Path],
    tenant_id: str,
    doc_id: Optional[str],
    data_dir: Path,
    config_dir: str,
    index_profile_name: str,
    skip_indexed: bool = True,
    force_reindex: bool = False,
    extra_tags: Optional[List[str]] = None,
) -> int:
    cfg = step1_load_config(config_dir)
    ingest_port = build_offline_ingest_port(cfg)
    index_profile = _parse_index_profile(index_profile_name)
    index_service, chroma_dir = create_offline_index_service(
        data_dir,
        cfg,
        config_dir=config_dir,
        index_profile=index_profile,
    )
    manifest = IndexManifest.for_data_dir(data_dir)
    log.info("Chroma 目录: %s", chroma_dir)
    log.info("索引清单: %s (skip_indexed=%s)", data_dir / "indexed_by_md5.json", skip_indexed)

    ok = 0
    skipped = 0
    failed = 0
    for i, path in enumerate(paths):
        if doc_id and len(paths) == 1:
            fid = doc_id
        else:
            fid = doc_id_from_file_md5(file_md5_hex(path))
        report = await build_one_document(
            path,
            fid,
            tenant_id,
            data_dir,
            config_dir,
            index_profile,
            cfg=cfg,
            ingest_port=ingest_port,
            index_service=index_service,
            chroma_dir=chroma_dir,
            manifest=manifest,
            skip_indexed=skip_indexed,
            force_reindex=force_reindex,
            extra_tags=extra_tags,
        )
        if report.skipped:
            skipped += 1
            ok += 1
        elif report.success:
            ok += 1
        else:
            failed += 1
            log.error("FAIL %s: %s", path.name, report.errors)

    built = ok - skipped
    log.info(
        "完成 成功=%d (新建=%d 跳过=%d) 失败=%d 总计=%d → %s",
        ok,
        built,
        skipped,
        failed,
        len(paths),
        chroma_dir,
    )
    return 0 if failed == 0 else 1


async def run_rebuild_bm25(
    data_dir: Path,
    config_dir: str,
    tenant_id: str,
) -> int:
    n, path = rebuild_bm25_from_chroma(
        data_dir,
        config_dir=config_dir,
        tenant_id=tenant_id,
    )
    log.info("BM25 重建完成: %d chunks → %s", n, path)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="离线 RAG 建库（六步流程见本文件顶部与 build_one_document）"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_SCAN_PATH),
        help=f"源文件或目录（默认 {DEFAULT_SCAN_PATH}）",
    )
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--doc-id", default=None)
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data" / "rag_offline"),
    )
    parser.add_argument(
        "--config-dir",
        default=str(REPO_ROOT / "config"),
    )
    parser.add_argument(
        "--index-profile",
        default="vector_only",
        choices=["vector_only", "sql_sidecar", "graph_sidecar", "full"],
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_SCAN_GLOB,
        help=f"目录扫描 glob（默认 {DEFAULT_SCAN_GLOB}）",
    )
    parser.add_argument(
        "--skip-indexed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="文件 MD5 已在 indexed_by_md5.json 中则跳过（默认开启）",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="忽略 MD5 清单，强制重新建库",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=None,
        dest="extra_tags",
        help="建库时附加 metadata 标签（可重复；与规则打标合并）",
    )
    parser.add_argument(
        "--rebuild-bm25",
        action="store_true",
        help="从 Chroma 全量重建 BM25 索引（不跑建库）",
    )
    args = parser.parse_args()

    if args.rebuild_bm25:
        sys.exit(
            asyncio.run(
                run_rebuild_bm25(
                    Path(args.data_dir),
                    args.config_dir,
                    args.tenant,
                )
            )
        )

    target = Path(args.path).resolve()
    log.info("扫描路径: %s  glob=%s", target, args.glob)
    files = _collect_files(target, args.glob)
    if not files:
        log.error("未找到文件: %s", target)
        sys.exit(1)

    sys.exit(
        asyncio.run(
            run_build(
                files,
                args.tenant,
                args.doc_id,
                Path(args.data_dir),
                args.config_dir,
                args.index_profile,
                skip_indexed=args.skip_indexed,
                force_reindex=args.force_reindex,
                extra_tags=args.extra_tags,
            )
        )
    )


if __name__ == "__main__":
    main()
