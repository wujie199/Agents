# document.rag — 建库 + 检索

分层：**bootstrap 组装 → application 用例 → components 可替换实现 → facades 对外 Port**

```text
document/rag/
├── config/                   # RagPipelineConfig + profile deep merge
├── bootstrap/
│   ├── offline.py            # 离线建库（Chroma + IndexService + 文件 embedding 缓存）
│   ├── online.py             # 在线 RAG（build_rag_stack）
│   └── query.py              # 离线查询栈
├── application/
│   ├── chunking/             # 七步分块 Step1–7
│   ├── embedding/            # 五步向量化 + 增量 + collection 解析
│   ├── indexing/             # IndexService / manifest / embedder
│   └── retrieval/            # hybrid + parent 扩展
├── components/               # ingest / embedding / storage / cleaner
├── facades/                  # RAGPort、KnowledgeBasePort
└── weights/                  # 本地模型权重
```

## 生产配置

```bash
export RAG_CONFIG=config/rag.production.example.yml
export RAG_EMBEDDING_MODEL_PATH=/opt/models/embedding/bge-small-zh-v1.5
export RAG_RERANK_MODEL_PATH=/opt/models/rerank/bge-reranker-base
export OCR_MODEL_ROOT=/opt/models/ocr
```

## 按文档类型 profile

| profile | 配置文件 | 适用 |
|---------|----------|------|
| `faq` | `config/rag.faq.yml` | FAQ PDF（OCR + 七步 + faq domain） |
| `contract` | `config/rag.contract.yml` | Word 合同（OCR + 七步 + legal/privacy 清洗） |

```bash
python document/build_rag_index.py --profile faq data/test_docs/*.pdf
python document/build_rag_index.py --profile contract contract.docx --glob "*.docx"
```

Web 上传 (`app/web/app.py`) 按扩展名：`.pdf` → `faq`，`.docx`/`.doc` → `contract`。

## 离线建库主流程

六步见 `document/build_rag_index.py`：

1. 读配置（基座 `rag.yml` + profile 增量）
2. 摄取（默认 `ocr_only`，Word 转 PDF 后 OCR）
3. 清洗（CompositeCleaner + OCR 后处理）
4. 规则 metadata 打标
5. 七步分块（`chunk_strategy: seven_step`）
6. 五步向量化 → Chroma + BM25

```bash
python document/build_rag_index.py data/test_docs --glob "*.pdf" --tenant default
# MD5 跳过：data/rag_offline/indexed_by_md5.json
# chunk 增量重编：--force-reindex（默认 chunk 级 diff，非全量 delete）
# 全量删除重建：embedding.force_full_delete_on_reindex: true
```

**增量与缓存**

| 项 | 说明 |
|----|------|
| 文件 MD5 manifest | 内容未变整文件跳过（`data/rag_offline/indexed_by_md5.json`） |
| chunk 指纹 | manifest 存 `chunk_fingerprints`，未变 chunk 不重编码 |
| embedding 文件缓存 | `data/rag_offline/embedding_cache/`（离线自动启用） |
| DLQ | 写入失败 → `embedding_dlq.jsonl`；重放见下方 |

```bash
# DLQ 重放（从 manifest 找源文件 force-reindex）
python scripts/replay_embedding_dlq.py --dry-run
python scripts/replay_embedding_dlq.py
```

**模型迁移**：`embedding.versioned_collection: true` 时 collection 为 `{base}_{model_slug}`，建库与检索均通过 `effective_collection_name()` 统一解析。

## 离线查询

```bash
python document/build_rag_index.py --rebuild-bm25 --data-dir data/rag_offline
python document/query_rag.py "扫地机器人如何回充"
python document/query_rag.py "退租流程" --scenario customer_service
```

检索命中子 chunk 时，若 metadata 含 `parent_id`，会自动从 `parent_store_dir` 扩展父 chunk 上下文。

BM25 路径：`data/rag_offline/bm25_index/{collection}.json`

## 在线 RAG

```python
from document.rag.bootstrap.online import build_rag_stack
```

## 摄取

| 能力 | 配置键 | 默认 |
|------|--------|------|
| 摄取 | `ingest.mode` | `ocr_only`（`structured` 等旧 mode 已废弃，回退 ocr_only） |
| 向量 | `embedding.backend` | `local_bge` |
| 重排 | `rerank.backend` | `local_bge` |
| 分块 | `chunk_strategy` | `seven_step` |
| 元数据 | `metadata.rules` | 规则关键词 |

切换 embedding 模型或分块配置后，需 `--force-reindex` 或更新 `config_hash` 触发重建。
