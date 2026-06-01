# document.rag — 建库 + 检索

分层：**bootstrap 组装 → application 用例 → adapters 可替换实现 → facades 对外 Port**

```text
document/rag/
├── config.py                 # RagPipelineConfig
├── bootstrap/                # ★ 唯一组装入口
│   ├── offline.py            # 离线建库（Chroma + IndexService）
│   └── online.py             # 在线 RAG（build_rag_stack）
├── adapters/                 # ★ 可替换实现 + registry
│   ├── registry.py           # build_* 工厂
│   ├── ingest/ | embedding/ | rerank/ | metadata/ | cleaning/
├── application/              # 用例编排（依赖 core.ports）
│   ├── indexing/ | ingest/ | cleaning/ | metadata/ | retrieval/
├── facades/                  # RAGPort、KnowledgeBasePort
├── shared/                   # 通用工具
└── weights/                  # 本地模型权重
```

`pipeline/`、`query/`、`bridges/`、`offline_stack.py` 为**兼容 re-export**，新代码请用上述路径。

## 替换适配器

编辑 `config/rag_pipeline.yml`，实现类在 `adapters/registry.py` 注册：

| 能力 | 配置键 | 默认 backend |
|------|--------|----------------|
| 摄取 | `ingest.mode` | `ocr_only` |
| 向量 | `embedding.backend` | `local_bge` |
| 重排 | `rerank.backend` | `local_bge` |
| 元数据打标 | `metadata.backend` | `rule_keyword`（规则见 `config/metadata_tagging.yml`） |

## 业务入口

**离线建库**（六步见 `document/build_rag_index.py`，含 [4] 规则打标）：

```bash
python document/build_rag_index.py document/rag/pdf --glob "*.pdf" --tenant default
# 默认按文件 MD5 跳过已索引（data/rag_offline/indexed_by_md5.json）
# 强制重建：--force-reindex
```

`doc_id` 默认 `doc_{md5前16位}`，与文件内容绑定，重复建库会 upsert 同文档向量。

切块策略默认 `chunk_strategy: faq`（一问一答），见 `application/indexing/faq_chunker.py`。
章节标题写入 metadata（`faq_category` / `faq_section`），不进入正文。

**在线 RAG**：

```python
from document.rag.bootstrap.online import build_rag_stack
# 或 core.composition.rag_factory_helpers.build_rag_stack（兼容）
```

**离线混合检索**（向量 + BM25 → 加权融合 → Rerank）：

```bash
# 首次启用 hybrid 且已有 Chroma、无 BM25 时，可先重建 BM25：
python document/build_rag_index.py --rebuild-bm25 --data-dir data/rag_offline

# 单次查询
python document/query_rag.py "扫地机器人如何回充"

# 按场景过滤（单库 agent + metadata.tags，见 config/scenarios.yml）
python document/query_rag.py "退租流程" --scenario customer_service

# 直接指定标签（可与 --scenario 叠加）
python document/query_rag.py "合同条款" --tag 合同 --tag 法律 --tag-match all

# 建库时附加手动标签（与规则打标合并）
python document/build_rag_index.py document/rag/pdf --tag 产品 --tag 营销

# 交互 REPL
python document/query_rag.py
```

**单库多场景**：所有文档写入同一 Chroma collection（默认 `agent`），通过 chunk metadata 的 `tags` 区分领域。建库时由 `config/metadata_tagging.yml` 规则打标，也可用 `--tag` 手动追加；检索时用 `--scenario` 或 `--tag` 在召回后过滤（向量/BM25 会过采样再过滤）。

配置见 `config/rag_pipeline.yml` → `retrieval`（`enable_hybrid`、`hybrid_weights`、`fusion_strategy`、`rerank_min_score`）。
Rerank 后仅保留分数 **大于** `rerank_min_score`（默认 0.8）的结果；关闭过滤可设 `rerank_min_score: null`。
BM25 索引路径：`data/rag_offline/bm25_index/{collection}.json`，建库时与 Chroma 同步写入。

**本地模型**：`weights/bge-small-zh-v1.5`（embedding）、`weights/bge-reranker-base`（rerank）。切换 backend 或维度后需重建 Chroma。
