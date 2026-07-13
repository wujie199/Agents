# RAG 迁移说明（向量 MVP）

## 范围

- **在范围内**：向量 RAG、混合检索（vector + BM25）、可选三库（vector / SQL / graph）、改写、Rerank、`IndexService`
- **配置开关**：`config/rag.yml`（+ profile 增量）中 `retrieval.enable_*` 与 `rewrite.enable_*`

## 主路径

```text
文件 → IngestFactory (ocr_only) → IndexService (seven_step) → Chroma + BM25
查询 → RAGPortAdapter → hybrid_retrieve 或 RetrievalRouter → EvidenceBundle
写入 → IndexService → Chroma（+ 可选 SQL/Graph sidecar，需 enable_graph_index）
```

## 配置

- 基座：`config/rag.yml`；profile：`config/rag.faq.yml`、`config/rag.contract.yml`
- 加载：`document/rag/config/` → `load_rag_pipeline_config()`
- 环境变量：`RAG_CONFIG`（兼容旧名 `RAG_PIPELINE_CONFIG`）
- Collection 默认 **`agent`**（`embedding.versioned_collection` 可后缀模型 slug）

## 模块

| 模块 | 路径 |
|------|------|
| 离线建库 | `document/build_rag_index.py` |
| 索引 | `document/rag/application/indexing/service.py` |
| 检索门面 | `document/rag/facades/rag.py` |
| 混合检索 | `document/rag/application/retrieval/hybrid_pipeline.py` |
| 摄取 | `document/rag/components/ingest/registry.py` |
| E2E | `scripts/rag_e2e.py` |

## 使用

```python
ctx = build_development_context(request)

from core.ports.index import IndexProfile
result = await ctx.require_knowledge_base().ingest_and_index(
    file_path, doc_id, tenant_id, index_profile=IndexProfile.VECTOR_ONLY,
)

bundle = await ctx.rag.route_and_retrieve(query, request)
```

```bash
python document/build_rag_index.py --profile faq data/test_docs/*.pdf
python document/query_rag.py "你的问题"
python scripts/rag_e2e.py path/to/doc.txt --query "你的问题"
```

## 多后端检索

见 **[RAG_BUSINESS_QUERY.md](./RAG_BUSINESS_QUERY.md)**。

- `retrieval.enable_router: true` → `RetrievalRouter`（生产 example）
- `enable_router: false` + `enable_hybrid: true` → hybrid（faq/contract 默认）

## Graph 二期

保留 `RetrievalRouter` / `GraphPort` 代码；基座与 profile 默认 `enable_graph_index: false`。
