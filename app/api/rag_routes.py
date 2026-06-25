# -*- coding: utf-8 -*-
"""RAG 知识库管理 HTTP 路由（文件上传 + 入库 + 文档列表 + 删除）。

挂载到 Chat API 的 /v1/rag/* 路径下。

启动后端后:
  curl -X POST http://localhost:8080/v1/rag/upload \
    -F "file=@doc.pdf" -F "tenant_id=tenant1"
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Coroutine, List, Optional

try:
    from fastapi import APIRouter, File, Form, HTTPException, UploadFile
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore[misc, assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    UploadFile = None  # type: ignore[misc, assignment]
    File = None  # type: ignore[misc, assignment]
    Form = None  # type: ignore[misc, assignment]
    BaseModel = object  # type: ignore[misc, assignment]

from core.ports.index import IndexProfile
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
from document.build_rag_index import build_one_document, step1_load_config


# ── 默认配置路径（由 chat_server 传入或使用默认值）──
_DEFAULT_CONFIG_DIR = "config"
_DEFAULT_DATA_DIR = str(Path("data") / "rag_offline")


class UploadResponse(BaseModel):
    success: bool
    doc_id: str = ""
    chunk_count: int = 0
    vectors_written: int = 0
    skipped: bool = False
    errors: list[str] = []


class DocumentInfo(BaseModel):
    doc_id: str
    source_path: str = ""
    chunk_count: int = 0
    vectors_written: int = 0
    indexed_at: str = ""


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    ok: bool
    doc_id: str


def create_rag_router(
    *,
    config_dir: str = _DEFAULT_CONFIG_DIR,
    data_dir: str = _DEFAULT_DATA_DIR,
    enforce_rate_limit: Optional[Callable[..., None]] = None,
    auth_dep: list = None,
) -> Any:
    """创建 RAG 管理路由器。"""
    if APIRouter is None:
        raise RuntimeError("RAG 路由需要安装: pip install fastapi")

    router = APIRouter(prefix="/v1/rag", tags=["RAG 知识库"])
    deps = auth_dep or []

    @router.post("/upload", response_model=UploadResponse, dependencies=deps)
    async def upload_document(
        file: UploadFile = File(...),
        tenant_id: str = Form(default="tenant1"),
        doc_id: Optional[str] = Form(default=None),
    ):
        """上传文件 → 摄取 → 切块 → 向量入库。"""
        if enforce_rate_limit:
            enforce_rate_limit(tenant_id, "system")

        data_path = Path(data_dir)
        cfg = step1_load_config(config_dir)
        ingest_port = build_offline_ingest_port(cfg)
        index_service, chroma_dir = create_offline_index_service(
            data_path, cfg, config_dir=config_dir, index_profile=IndexProfile.VECTOR_ONLY
        )
        manifest = IndexManifest.for_data_dir(data_path)

        # 将上传文件保存到临时目录
        suffix = Path(file.filename or "upload.bin").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            fid = doc_id or doc_id_from_file_md5(file_md5_hex(tmp_path))
            report = await build_one_document(
                tmp_path,
                fid,
                tenant_id,
                data_path,
                config_dir,
                IndexProfile.VECTOR_ONLY,
                cfg=cfg,
                ingest_port=ingest_port,
                index_service=index_service,
                chroma_dir=chroma_dir,
                manifest=manifest,
                skip_indexed=True,
                force_reindex=False,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        if report.success:
            return UploadResponse(
                success=True,
                doc_id=report.doc_id,
                chunk_count=report.index.chunk_count if report.index else 0,
                vectors_written=report.index.vectors_written if report.index else 0,
                skipped=report.skipped,
            )
        return UploadResponse(
            success=False,
            doc_id=fid,
            errors=report.errors or ["ingest failed"],
        )

    @router.get("/documents", response_model=DocumentListResponse, dependencies=deps)
    async def list_documents(tenant_id: str = "tenant1"):
        """列出已索引文档。"""
        manifest = IndexManifest.for_data_dir(Path(data_dir))
        data = manifest._load()
        entries = data.get("tenners", {}).get(tenant_id, {})
        # fallback: data 结构可能是 tenants
        if not entries:
            entries = data.get("tenants", {}).get(tenant_id, {})

        docs = []
        for md5, entry in entries.items():
            docs.append(DocumentInfo(
                doc_id=entry.get("doc_id", ""),
                source_path=entry.get("source_path", ""),
                chunk_count=entry.get("chunk_count", 0),
                vectors_written=entry.get("vectors_written", 0),
                indexed_at=entry.get("indexed_at", ""),
            ))
        return DocumentListResponse(documents=docs, total=len(docs))

    @router.delete("/documents/{doc_id}", response_model=DeleteResponse, dependencies=deps)
    async def delete_document(doc_id: str, tenant_id: str = "tenant1"):
        """删除已索引文档。"""
        data_path = Path(data_dir)
        cfg = step1_load_config(config_dir)
        _, _ = create_offline_index_service(
            data_path, cfg, config_dir=config_dir, index_profile=IndexProfile.VECTOR_ONLY
        )
        # IndexService.delete_document 由 index_service 实例方法调用
        # 这里通过 manifest 移除条目
        manifest = IndexManifest.for_data_dir(data_path)
        data = manifest._load()
        tenants = data.get("tenants", {})
        tenant_entries = tenants.get(tenant_id, {})
        # 找到匹配 doc_id 的条目并删除
        removed = False
        for md5, entry in list(tenant_entries.items()):
            if entry.get("doc_id") == doc_id:
                del tenant_entries[md5]
                removed = True
                break
        if removed:
            manifest._save(data)
        return DeleteResponse(ok=removed, doc_id=doc_id)

    return router
