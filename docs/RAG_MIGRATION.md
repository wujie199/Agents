# RAG 迁移说明（向量 MVP）

## 范围

- **在范围内**：向量 RAG、三库检索（vector / SQL / graph）、改写（HyDE / Multi-Query）、Rerank、`IndexService`、`RetrievalRouter`
- **配置开关**：`config/rag_pipeline.yml` 中 `retrieval.enable_*` 与 `rewrite.enable_*`

## 主路径

```text
文件 → IngestFactory → IndexService → Chroma (collection: agent)
查询 → RAGPortAdapter → RetrievalRouter（改写 → 三库 → 融合 → rerank）→ EvidenceBundle
写入 → IndexService → Chroma + documents 表（SQL）+ 图节点（Graph，需 enable_graph_index）
```

## 配置

- 统一配置：`config/rag_pipeline.yml`
- 加载器：`document/rag/config.py` → `load_rag_pipeline_config()`
- Collection 名称默认 **`agent`**（与 `config/chroma.yml` 对齐）

## 模块

| 模块 | 路径 |
|------|------|
| 索引 | `document/rag/pipeline/index/service.py` |
| 检索 | `document/rag/bridges/rag_port_adapter.py` |
| 摄取路由 | `document/rag/pipeline/ingest/factory.py` |
| 开发缓存 | `agent_platform/storage/adapters/memory/async_cache_adapter.py` |
| E2E | `scripts/rag_e2e.py` |

## 使用

```python
ctx = build_development_context(request)

# 推荐：建库门面（文件 → 摄取 → 索引）
from core.ports.index import IndexProfile
result = await ctx.require_knowledge_base().ingest_and_index(
    file_path, doc_id, tenant_id, index_profile=IndexProfile.VECTOR_ONLY,
)

# 或仅索引已有文本
await ctx.require_index().index_document(doc_id, content, tenant_id)

bundle = await ctx.rag.route_and_retrieve(query, request)
```

```bash
python scripts/rag_e2e.py path/to/doc.txt --query "你的问题"
```

## 多后端检索（业务怎么查）

见 **[RAG_BUSINESS_QUERY.md](./RAG_BUSINESS_QUERY.md)**：业务只调 `ctx.rag.route_and_retrieve(query, request, plan?)`；`plan` 由 Router Agent 下发或留空自动分类。

- `retrieval.enable_router: true` 时 `RAGPortAdapter` 委托 `RetrievalRouter`
- `enable_graph` / `enable_sql` 控制是否启用图/SQL（需组合根注入对应 Port）

## Graph 二期

保留 `RetrievalRouter` / `GraphPort` 代码，组合根不注入图写入；`enable_graph_index: false`。
