# 企业级记忆 + Chat 接入

## 0. 开发默认（开箱即用）

```bash
python app/chat_repl.py --tenant tenant1 --user user1 --session chat1
```

- **记忆**：`config/memory.yml` 分段配置（l1/l2/archive/vector/skills/l4/cold_archive）；dev 默认全开，**无需** `MEMORY_CONFIG`
- 联调 profile：`MEMORY_PROFILE=vector|cold|skills|l4_http`
- **启动 bootstrap**：自动创建目录、L4 seed、ensure_session、当前会话向量 reindex
- **Chat dev profile**（`config/chat.yml` → `profiles.dev`）：`remember_require_hitl: false`；退出 REPL 时 `auto_confirm_pending_on_exit: true` 自动确认剩余 pending
- **RAG 租户**：见 [TENANT.md](./TENANT.md)

REPL 额外命令：`/refresh-profile`（L4 缓存刷新）、`/debug-memory`（需 `--debug` 或 `MEMORY_RUNTIME_DEBUG=1` 写 trace）。

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
| L4 画像刷新 | `/refresh-profile` | `POST /v1/memory/l4/refresh` |
| 运行状态 | `/debug-memory` | `GET /v1/memory/runtime` |
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
| `GET /metrics` | Prometheus 文本（`memory.*`、`graph.*`、`agent.*` 计数器） |
| `GET /v1/metrics/memory` | JSON 指标摘要（count/sum/avg） |
| `POST /v1/memory/status` | 含 `metrics` 字段 |
| REPL `/metrics` | 同上 JSON + Prometheus 片段 |

### 指标

| 指标 | 说明 |
|------|------|
| `graph.node.duration_ms` | LangGraph 节点耗时（tags: `node`, `tenant_id`, `error`, `slow`） |
| `graph.node.errors_total` | 节点异常计数（tags: `node`, `error_type`, `tenant_id`） |
| `agent.tool.*` / `agent.llm.*` | ReAct tool/LLM 调用 |
| `memory.*` / `cache.rag.*` / `cache.redis.*` | 记忆与缓存 |

`error_type` 取值：`policy_denied`、`llm_timeout`、`tool_error`、`memory_error`、`unknown`。

### Trace 流

```
HTTP RequestContext → Middleware 链 → LangGraph 节点
  RequestContextMiddleware  解析 trace_id / tenant / user / session
  TracingMiddleware         graph.{node} span（ObservabilityPort / OTel）
  TimingMiddleware          慢节点标记
  MetricsMiddleware         graph.node.duration_ms
  PolicyMiddleware          租户 QPS（内存或 Redis 滑动窗口）
  PrivacyMiddleware         PII 检测
  ErrorClassifierMiddleware graph.node.errors_total（on error）
  AuditMiddleware           hash 审计 + 可选 NDJSON 持久化
ReAct agent 节点：astream_events(v2) → agent.tool.* / agent.llm.*
```

LangGraph 节点经 Middleware 链注入 `trace_id` 与 span 属性（`tenant_id` / `user_id` / `session_id` / `node`）。ReAct agent 节点通过 `astream_events(v2)` 记录 tool/LLM 耗时，并写入 `run_ctx.extra.agent_events` 供 SSE meta 使用。

### 配置（config/chat.yml → observability）

```yaml
observability:
  enabled: true
  trace_header: X-Request-ID
  slow_threshold_ms:
    prepare: 2000
    agent: 8000
    persist: 500
  rate_limit_backend: memory   # memory | redis
  max_qps_per_tenant: 100
  audit_persist: false         # true → data/audit/audit_YYYY-MM-DD.jsonl
  audit_log_dir: data/audit
```

- **Redis 限流**：设置 `observability.rate_limit_backend: redis` 或环境变量 `REDIS_URL` 时，PolicyMiddleware 使用 Redis 1 秒滑动窗口；Redis 不可用时 **fail-open** 回退进程内限流。
- **审计持久化**：`audit_persist: true` 时追加 NDJSON，字段含 `trace_id`、`tenant_id`、`user_id`、`session_id`、`node`、`duration_ms`、`error`、`content_hashes`、`ts`。

### OpenTelemetry（可选）

未安装 OTel 包或未设置 endpoint 时自动回退内存 adapter，不影响服务启动。

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
export OTEL_SERVICE_NAME=agents-chat          # 默认 agents-chat
export OTEL_TRACES_SAMPLER_ARG=0.1            # head sampling，1.0=全采样
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

`production_factory` / `chat_server` 在检测到 `OTEL_EXPORTER_OTLP_ENDPOINT` 时使用 `OtelObservabilityAdapter`，否则 `ObservabilityPortAdapter`。

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
