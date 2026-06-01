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

`build_production_context` 默认用 Redis 缓存 `session_search` 摘要；开发 CLI 用内存缓存。  
生产需保证 Redis 可用，否则检索仍可用但无跨实例缓存。

## 8. 尚未接入 Agent 时

PG 生产只保证 **L2 归档读写 + CLI**；对话自动 `persist_turn` 仍须后续接 Agent。  
上线前可用 `session-append` 做冒烟写入。
