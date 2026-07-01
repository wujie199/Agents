# L2 Session Archive：SQLite → PostgreSQL 迁移

## 前提

- 已安装 `asyncpg`，PostgreSQL 可连接
- 目标库已创建（默认库名 `agents`，见 `config/memory.yml`）
- **向量索引在 PG 中为空**；迁移后需单独执行 `reindex`（见下文）

## 配置

1. 在 `config/memory.yml` 中配置 PG 连接（`pg_host`、`pg_database`、`pg_user` 等），或使用环境变量 `PGHOST`、`PGPASSWORD` 等。
2. 迁移阶段**不必**先把 `archive_backend` 改为 `postgresql`；CLI 会显式连接源 SQLite 与目标 PG。

## 迁移命令

```bash
# 预览：只统计行数，不写 PG
python document/query_memory.py migrate-archive \
  --sqlite-path data/memory_cli.db \
  --dry-run

# 执行迁移（默认源：config 中 archive_sqlite_path 或 data/memory_cli.db）
export PGPASSWORD=your_secret
python document/query_memory.py migrate-archive \
  --sqlite-path data/session_archive.db

# 只迁移某租户 / 用户
python document/query_memory.py migrate-archive \
  --sqlite-path data/session_archive.db \
  --tenant tenant1 --user user1

# 迁移完成后补向量索引（需已 enable_session_vector_index 或 --vector）
python document/query_memory.py --vector reindex \
  --tenant tenant1 --user user1

# 一键：迁移 + reindex
python document/query_memory.py migrate-archive \
  --sqlite-path data/session_archive.db \
  --reindex --vector
```

## 行为说明

| 表 | 策略 |
|---|---|
| `sessions` | `INSERT ... ON CONFLICT DO NOTHING` |
| `messages` | `INSERT ... ON CONFLICT DO UPDATE`（幂等，可重复跑） |
| `tool_calls` | `INSERT ... ON CONFLICT DO NOTHING` |

- **不迁移** SQLite 的 `messages_fts`；PG 使用 `to_tsvector` GIN 索引，数据写入后即可检索。
- **不自动迁移** L1 热记忆目录（`store_dir` 下 `MEMORY.md` / `USER_*.md`），需自行复制或保留原路径。

## 切换生产归档库

迁移并验证检索无误后：

```yaml
# config/memory.yml
archive_backend: postgresql
```

重启服务或 CLI 即默认读写 PostgreSQL。

## 验证

```bash
python document/query_memory.py session-list --tenant tenant1 --user user1
python document/query_memory.py session-search \
  --tenant tenant1 --user user1 --session <id> --query "关键词"
```

## 回滚

保留原 SQLite 文件备份；将 `archive_backend` 改回 `sqlite` 并指向原 `archive_sqlite_path` 即可回退只读旧库（勿双写）。

完整上线清单见 [PG_PRODUCTION.md](./PG_PRODUCTION.md)。
