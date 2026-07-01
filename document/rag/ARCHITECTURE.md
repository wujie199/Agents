# RAG 模块架构

## 分层设计

```
document/rag/
├── bootstrap/       # ★ 唯一组装入口
├── application/     # 用例编排层
├── adapters/        # 可替换实现层
├── facades/         # 对外 Port 层
└── shared/          # 通用工具
```

## 目录职责

| 目录 | 职责 | 独立部署 |
|------|------|---------|
| **bootstrap/** | 组装入口：`online.py`（在线 RAG）、`offline.py`（离线建库）、`query.py`（查询） | ✅ |
| **application/** | 用例编排：`indexing/`、`retrieval/`、`ingest/`、`cleaning/`、`metadata/` | ✅ |
| **adapters/** | 可替换实现：`embedding/`、`rerank/`、`ingest/`、`cleaning/`、`metadata/`、`retrieval/` | ✅ |
| **facades/** | 对外 Port 实现：`RAGPortAdapter`、`KnowledgeBasePortAdapter` | ✅ |
| **shared/** | 通用工具：文本清洗、分块、文件处理 | ✅ |

## 核心路径

- 新代码使用：`bootstrap/` → `application/` → `adapters/` → `facades/` → `shared/`
