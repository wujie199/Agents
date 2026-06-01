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
