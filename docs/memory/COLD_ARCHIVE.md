# L2 冷归档

过期或手动指定的 L2 会话会从在线归档库（SQLite / PostgreSQL）导出为 JSON（可选 gzip），写入对象存储，并在 `cold_archive_sessions` 表建立索引，随后删除在线 `sessions` / `messages` / `tool_calls` 明细。

## 配置（`config/memory.yml`）

```yaml
enable_cold_archive: false   # 生产示例见 memory.production.example.yml
cold_archive_prefix: l2/cold
cold_archive_compress: true
cold_archive_bucket: agents-storage
retention_days: 90
```

启用后需配置 `object_store`（生产为 S3/OBS；开发无凭证时 `S3ObjectStoreAdapter` 降级到 `data/objects/`）。

## CLI（`document/query_memory.py`）

```bash
# 启用冷归档（或 config 中 enable_cold_archive: true）
python document/query_memory.py --cold-archive archive-session \
  --tenant t1 --user u1 --session sess-001

python document/query_memory.py --cold-archive archive-expired --days 90

python document/query_memory.py --cold-archive cold-list --tenant t1 --user u1

python document/query_memory.py --cold-archive cold-fetch \
  --tenant t1 --user u1 --session sess-001
```

`purge-expired` 在冷归档已配置时会自动走冷归档路径（等价 `archive-expired`），否则硬删除在线行。

## 对象路径

```
{cold_archive_prefix}/{tenant_id}/{user_id}/{session_id}.json.gz
```

## 合规

`purge_user_data` 会同时删除该用户的冷归档对象与索引行。

## 检索

`session_search` 在线库无结果时会 fallback 检索冷归档对象（`session_search_cold_fallback: true`）。

## 生产运维

### Cron（过期冷归档）

```bash
# 每日 03:00 归档过期会话
0 3 * * * cd /path/to/Agents && python scripts/memory_archive_cron.py --cold-archive --days 90 --pg

# 每周补建冷归档 DB 检索索引（归档后历史数据）
0 4 * * 0 cd /path/to/Agents && python scripts/memory_archive_cron.py \
  --cold-archive --backfill-cold-search --tenant YOUR_TENANT --pg
```

### 历史数据 backfill

```bash
# 冷搜索索引 + 向量 reindex 一次完成（需 --cold-archive --vector）
python document/query_memory.py --cold-archive --vector backfill-all \
  --tenant t1 [--user u1] [--dry-run]

# 分项
python document/query_memory.py --cold-archive backfill-cold-search --tenant t1
python document/query_memory.py --vector reindex --tenant t1
```

生产配置见 `config/memory.production.example.yml`（PG、`enable_cold_archive: true`、`enable_session_vector_index: true`）。

## Checkpointer

图状态快照与 L2 分工见 `document/memory/CHECKPOINTER.md`。
