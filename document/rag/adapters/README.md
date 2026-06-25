# Adapters — 可替换实现层

通过 `registry.py` 注册的适配器实现，支持热替换。

## 子模块

| 子目录 | 职责 |
|--------|------|
| `embedding/` | 向量 Embedding：本地 BGE / Mock |
| `rerank/` | Rerank 模型：本地 BGE Reranker / Mock |
| `ingest/` | 文档摄取：Word、PDF (OCR)、纯文本 |
| `cleaning/` | 文档清洗：组合清洗器、领域清洗器 |
| `metadata/` | 元数据 enrichment：规则关键词 / No-Op |
| `retrieval/` | 检索后端：BM25 本地索引 |
