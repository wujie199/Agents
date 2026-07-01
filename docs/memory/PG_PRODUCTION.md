# L2 上 PostgreSQL 生产 — 上线清单

> 迁移细节见 [MIGRATE_ARCHIVE.md](./MIGRATE_ARCHIVE.md)

## 1. 依赖与库

```bash
pip install asyncpg chromadb   # 若启用向量
createdb agents                # 或 RDS 上建库
```

应用账号建议**仅** L2 表权限（sessions / messages / tool_calls），与 RAG 库分离。

## 2. 配置

```bash
# 方式 A：专用生产配置
export MEMORY_CONFIG=config/memory.production.example.yml
export PGPASSWORD='...'

# 方式 B：DATABASE_URL（覆盖 pg_host 等）
export DATABASE_URL='postgresql://agents_app:secret@db-host:5432/agents'
export MEMORY_CONFIG=config/memory.production.example.yml
```

`config/memory.production.example.yml` 中已设 `archive_backend: postgresql`。  
**勿**在仓库提交真实密码。

## 3. 健康检查

```bash
python document/query_memory.py --pg archive-health
# 或 MEMORY_CONFIG 已指向生产 yml 时：
python document/query_memory.py archive-health
```

期望：`archive_db.status=healthy`，且 `sessions/messages/tool_calls` 表存在。

## 4. 从 SQLite 迁数据（若有历史）

```bash
python document/query_memory.py migrate-archive \
  --sqlite-path data/session_archive.db \
  --dry-run

export PGPASSWORD=...
python document/query_memory.py migrate-archive \
  --sqlite-path data/session_archive.db \
  --reindex --vector --tenant YOUR_TENANT
```

## 5. 切流量

1. 确认 `MEMORY_CONFIG` / `memory.yml` 中 `archive_backend: postgresql`
2. 重启 `build_production_context` 所在服务（已走 `build_archive_db`）
3. CLI 验证：

```bash
python document/query_memory.py session-list --tenant t1 --user u1
python document/query_memory.py session-search \
  --tenant t1 --user u1 --session s1 --query "测试"
```

## 6. 运维

| 任务 | 命令 / 说明 |
|---|---|
| 过期清理 | `python document/query_memory.py purge-expired --days 90`（建议 cron） |
| 补向量 | `python document/query_memory.py --vector reindex --tenant t1` |
| 合规删除 | `purge-user --tenant t1 --user u1` |
| 回滚 | `archive_backend: sqlite` + 保留的 `.db` 备份（勿双写） |

## 7. Redis（生产）

dev / production **均使用 Redis** 缓存 `session_search` 摘要与 RAG 检索结果；L4 画像缓存前缀 `l4:`。  
限流默认使用 `REDIS_URL` 的 **db/1**（缓存 db/0），也可用 `CHAT_RATE_LIMIT_REDIS_URL` 单独指定。

```bash
export REDIS_URL=redis://localhost:6379/0
# 可选: export CHAT_RATE_LIMIT_REDIS_URL=redis://localhost:6379/1
```

`GET /health` 探测 Redis + L2；`GET /ready`（production strict）要求依赖全部 healthy。

## 8. 对话写入

Chat / REPL 已自动 `persist_turn`；上线前仍建议 `session-append` 或一轮对话冒烟。

## 9. Checkpointer 清理

```bash
python document/query_memory.py checkpoint-purge --days 90
# 或 cron: python scripts/memory_archive_cron.py --checkpoint-purge
```
