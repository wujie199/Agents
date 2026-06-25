# Application — 用例编排层

RAG 业务用例的编排实现。

## 子模块

| 子目录 | 职责 |
|--------|------|
| `indexing/` | 文档索引：分块、Embedding、入库、图同步 |
| `retrieval/` | 检索：混合管道、Rerank、Query Rewrite、Router |
| `ingest/` | 文档摄取工厂 |
| `cleaning/` | 文档清洗管道 |
| `metadata/` | 元数据 enrichment 管道 |
