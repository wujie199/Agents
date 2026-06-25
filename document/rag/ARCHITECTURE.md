# RAG 模块架构

## 分层设计

```
document/rag/
├── bootstrap/       # ★ 唯一组装入口
├── application/     # 用例编排层
├── adapters/        # 可替换实现层
├── facades/         # 对外 Port 层
├── shared/          # 通用工具
├── pipeline/        # [兼容] 离线管道 re-export
├── query/           # [兼容] 在线检索 re-export
└── bridges/         # [兼容] 桥接 re-export
```

## 目录职责

| 目录 | 职责 | 独立部署 |
|------|------|---------|
| **bootstrap/** | 组装入口：`online.py`（在线 RAG）、`offline.py`（离线建库）、`query.py`（查询） | ✅ |
| **application/** | 用例编排：`indexing/`、`retrieval/`、`ingest/`、`cleaning/`、`metadata/` | ✅ |
| **adapters/** | 可替换实现：`embedding/`、`rerank/`、`ingest/`、`cleaning/`、`metadata/`、`retrieval/` | ✅ |
| **facades/** | 对外 Port 实现：`RAGPortAdapter`、`KnowledgeBasePortAdapter` | ✅ |
| **shared/** | 通用工具：文本清洗、分块、文件处理 | ✅ |
| **pipeline/** | 离线管道兼容 re-export（指向 `application/` + `adapters/`） | ❌ |
| **query/** | 在线检索兼容 re-export（指向 `application/retrieval/`） | ❌ |
| **bridges/** | 桥接兼容 re-export（指向 `facades/` + `adapters/`） | ❌ |

## 核心路径

- 新代码使用：`bootstrap/` → `application/` → `adapters/` → `facades/` → `shared/`
- `pipeline/`、`query/`、`bridges/` 为向后兼容层，逐步迁移至新路径
