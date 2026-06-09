# L8 HTTP 接入层

## Chat API

可选依赖：

```bash
pip install fastapi uvicorn
# 或
pip install -e ".[api,langgraph]"
```

启动：

```bash
# 开发 profile（默认）
uvicorn app.api.chat_server:app --reload --port 8080

# 生产 profile
python -m app.api.chat_server --port 8080 --profile production
```

### 鉴权（可选）

```bash
export CHAT_API_KEY=your-secret
curl -H 'Authorization: Bearer your-secret' ...
# 或
curl -H 'X-API-Key: your-secret' ...
```

未设置 `CHAT_API_KEY` 时不启用鉴权（仅适合本地开发）。

### 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 依赖健康检查（Redis / L2 / 冷归档 / Checkpointer） |
| GET | `/ready` | K8s readiness（production 严格模式） |
| GET | `/metrics` | Prometheus 文本（`memory.*`） |
| GET | `/v1/metrics/memory` | 记忆指标 JSON 摘要 |
| POST | `/v1/chat/turn` | 单轮对话（JSON） |
| POST | `/v1/chat/turn/stream` | 单轮对话（SSE 流式） |
| POST | `/v1/chat/snapshot` | L1 热记忆快照 |
| POST | `/v1/chat/pending` | 待确认 L1 记忆（HITL） |
| POST | `/v1/chat/end` | 结束会话（finalize） |

### 企业记忆管理 `/v1/memory/*`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/memory/config` | 记忆配置摘要 |
| POST | `/v1/memory/status` | L1/L2 状态 |
| POST | `/v1/memory/sessions` | L2 会话列表 |
| POST | `/v1/memory/pending` | pending L1 |
| POST | `/v1/memory/confirm` | 确认 pending L1 |
| POST | `/v1/memory/purge/user` | 删除用户数据（需 `confirm=true`） |
| POST | `/v1/memory/purge/tenant/l3` | 删除租户 L3 技能数据（需 `confirm=true`） |
| POST | `/v1/memory/purge/tenant/l4` | 删除租户 L4 画像（需 `confirm=true`） |
| POST | `/v1/memory/retention/run` | 过期 session 清理 |
| POST | `/v1/memory/checkpoint/purge` | 清理超期 graph_checkpoints |

详见 `document/memory/ENTERPRISE_CHAT.md`。

### 示例

```bash
curl -s http://127.0.0.1:8080/v1/chat/turn \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"tenant1","user_id":"user1","session_id":"s1","message":"介绍一下扫地机器人"}'

curl -N http://127.0.0.1:8080/v1/chat/turn/stream \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"tenant1","user_id":"user1","session_id":"s1","message":"你好","engine":"langgraph"}'

curl -s http://127.0.0.1:8080/v1/chat/end \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"tenant1","user_id":"user1","session_id":"s1"}'
```

`engine` 可选 `langgraph`（默认）或 `direct`（含 Path B ReAct 工具）。

### 流式 `stream_mode`

| 值 | 行为 |
|---|---|
| `auto` | LangGraph → `astream_events` token 流；direct 无工具 → token 流；否则整轮后分块 |
| `token` | 优先 token 级流式 |
| `batch` | 整轮完成后分块推送 |

### 限流

配置：`config/concurrency.yml` → `chat`（默认 60/min/tenant，20/min/user）。

分布式限流（多实例）：

```bash
export CHAT_RATE_LIMIT_REDIS_URL=redis://localhost:6379/0
# 或 REDIS_URL=...
```

未配置 Redis 时使用进程内滑动窗口。

### 环境变量

| 变量 | 说明 |
|------|------|
| `REDIS_URL` / `REDIS_HOST` | **dev/production 均使用 Redis 缓存**（默认 `localhost:6379`；前缀 `CHAT_CACHE_REDIS_PREFIX`，默认 `agents`） |
| `CHAT_RATE_LIMIT_REDIS_URL` | 限流 Redis（未设时用 `REDIS_URL` 的 db/1，与缓存 db/0 分离） |
| `USE_MEMORY_GRAPH` | production：`false` 用 Neo4j（默认 false）；`true` 用内存图 |
| `MEMORY_CONFIG` | 记忆配置路径；production profile 未设时自动用 `memory.production.example.yml` |
| `MEMORY_ADMIN_API_KEY` | purge/retention 专用 Key（默认同 `CHAT_API_KEY`） |
| `RAG_TENANT_ID` | RAG 检索 tenant（默认 `default`） |
| `CHAT_API_KEY` | API Key（可选） |
| `CHAT_RATE_LIMIT_REDIS_URL` | 分布式限流 Redis URL（可选） |
