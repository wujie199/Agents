# 企业级记忆 + Chat 接入

## 1. 生产配置

```bash
export MEMORY_CONFIG=config/memory.production.example.yml
# 或仅 --profile production（自动指向 memory.production.example.yml）
export DATABASE_URL='postgresql://agents_app:secret@localhost:5432/agents'  # 可选
export REDIS_URL=redis://localhost:6379/0   # 可选，默认 localhost:6379；dev/production 均走 Redis 缓存
export CHAT_API_KEY=your-secret
export LANGGRAPH_CHECKPOINT_PATH=data/langgraph_checkpoints.db

python -m app.api.chat_server --profile production --port 8080
```

Chat Agent 配置（可选）：

```bash
# 复制 chat.production.example.yml 并调整
```

## 2. Path B 记忆工具（Chat Agent）

`enable_memory_tools=true` 时注册：

| 工具 | 层 | 说明 |
|------|-----|------|
| `session_search` | L2 | scope=session/user 跨会话回忆 |
| `remember_user_fact` | L1 | HITL 可选 |
| `skill_search` / `run_skill` | L3 | `enable_skill_tools` |
| `resolve_entity` / `fetch_profile_facts` | L4 | `enable_l4_tools` |

Context 预检索：`session_search_prefetch_scope`（auto/session/user）、`skill_prefetch`、`l4_profile_prefetch`。

## 3. 能力矩阵

| 能力 | CLI | HTTP |
|------|-----|------|
| L1 快照 | `/snapshot` | `POST /v1/chat/snapshot` |
| pending L1 | `/pending` | `POST /v1/memory/pending` |
| 确认 L1 | `/confirm` | `POST /v1/memory/confirm` |
| L2 会话列表 | `/sessions` | `POST /v1/memory/sessions` |
| 记忆状态 | `/status` | `POST /v1/memory/status` |
| 用户数据删除 | — | `POST /v1/memory/purge/user` |
| 租户 L3/L4 删除 | CLI | `POST /v1/memory/purge/tenant/l3`、`/l4` |
| 过期清理 | cron | `POST /v1/memory/retention/run` |
| Checkpoint 清理 | CLI | `POST /v1/memory/checkpoint/purge` |
| 配置摘要 | — | `GET /v1/memory/config` |
| 记忆指标 | `/metrics` | `GET /v1/metrics/memory`、`GET /metrics` |

## L1 存储

| `l1_store_backend` | 说明 |
|---------------------|------|
| `file` | 默认；`l1_use_file_lock` 支持多 Pod 共享 PVC |
| `relational` | PG/SQLite 同库 `hot_memory_docs` 表（生产 PG 推荐） |

## 4. Checkpointer

- **L2 Session Archive**：messages / tool_calls（PG/SQLite，合规真相源）
- **LangGraph Checkpointer**：
  - **dev / SQLite L2**：`data/langgraph_checkpoints.db`（与 L2 同目录）
  - **production / PostgreSQL L2**：`AsyncPostgresSaver` 与 L2 **共用 `DATABASE_URL`**（启动时 `resolve_chat_checkpointer_async` 自动初始化）
- Chat 每轮从 L2 重建 messages，Checkpointer 用于图状态 / 中断恢复
- 进程退出时调用 `teardown_postgres_checkpointer()` 释放 PG 连接

## 5. 可观测性

| 端点 | 说明 |
|------|------|
| `GET /metrics` | Prometheus 文本（`memory.*` 计数器） |
| `GET /v1/metrics/memory` | JSON 指标摘要（count/sum/avg） |
| `POST /v1/memory/status` | 含 `metrics` 字段 |
| REPL `/metrics` | 同上 JSON + Prometheus 片段 |

指标名：`memory.*`、`cache.rag.*`、`cache.redis.*`（含 circuit breaker / fallback）

## 6. 健康检查

| 端点 | 说明 |
|------|------|
| `GET /health` | Redis、L2 Archive、冷归档、Checkpointer |
| `GET /ready` | production 严格模式（依赖未就绪返回 503） |

CLI：`python document/query_memory.py archive-health`、`checkpoint-health`

## 7. 检索 LLM 路由（可选）

`config/chat.yml` 中 `retrieval_llm_router: true` 启用 `router_llm` 意图分类；低置信度回退正则。

## 8. 定时任务

```bash
# L2 冷归档 / 过期清理
python scripts/memory_archive_cron.py --cold-archive --days 90
python scripts/memory_archive_cron.py --days 90  # 仅 purge 过期
```

## 9. 上线清单

见 [PG_PRODUCTION.md](./PG_PRODUCTION.md)、[COLD_ARCHIVE.md](./COLD_ARCHIVE.md)。
