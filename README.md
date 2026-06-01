# Agents — 多 Agent 协作底座

## 目录结构

| 顶层 | 说明 |
|------|------|
| `core/` | 契约与组装：`domain/`、`ports/`、`composition/` |
| `document/` | 文档链路：`ocr/`（解析）、`rag/`（摄取/索引/检索） |
| `agent_platform/` | 平台能力：存储、LLM 注册、记忆、工具（非 `platform`，避免与标准库冲突） |
| `app/` | 预留：`agents/`、`workflows/`、`runtime/`、`api/` |
| `assets/` | 模型权重（`llm/`、`vision/`），无 Python 代码 |
| `config/` | YAML 配置 |
| `utils/` | 通用工具 |
| `tests/`、`scripts/`、`data/`、`docs/` | 测试、脚本、运行数据、文档 |

- OCR：`document/ocr/README.md`
- RAG：`document/rag/README.md`
- 平台：`agent_platform/README.md`

## RAG 三行用法

```python
from core.domain.context import RequestContext
from core.composition.production_factory import build_development_context
from core.ports.index import IndexProfile

request = RequestContext(tenant_id="t1", user_id="u1", session_id="s1", trace_id="tr1", channel="cli")
ctx = build_development_context(request)

await ctx.require_knowledge_base().ingest_and_index("doc.txt", "doc1", "t1", index_profile=IndexProfile.VECTOR_ONLY)
bundle = await ctx.require_rag().route_and_retrieve("你的问题", request)
```

CLI：`python scripts/rag_e2e.py path/to/file.txt`

## 离线 RAG 建库

```bash
conda activate py3.11
python document/build_rag_index.py path/to/doc.pdf --tenant t1 --doc-id doc1
```

流程：摄取（OCR）→ 格式清理 → 切块 → Embedding → Chroma（`data/rag_offline/chroma_dev`）。

## OCR 试跑

```bash
conda activate py3.11
python document/ocr/processor.py document/ocr/test.pdf --max-pages 5
```

## 文档

- 架构：`docs/ARCHITECTURE.md`
- 实施计划：`docs/NEXT_STEPS.md`
- RAG：`docs/RAG_MIGRATION.md`、`docs/RAG_BUSINESS_QUERY.md`
