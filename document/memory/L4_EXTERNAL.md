# L4 External Provider（外部画像）

## 目录

| 路径 | 用途 |
|------|------|
| `data/external_profiles/{tenant}/{user}.yaml` | 实体别名 + 结构化 facts（file 后端） |

## YAML 格式

```yaml
entities:
  张三:
    canonical_id: u001
    display_name: 张三

facts:
  - key: 部门
    value: 研发部
    source: ldap
  - key: 职位
    value: 高级工程师
    source: hr
```

## 与 L1 的协作

- L4 facts **不**整表写入 prompt；在 `finalize-session`（或 `end_session(finalize=True)`）时合并进 L1 `USER_{user}.md`
- 同 key 冲突时 **L4 覆盖** L1 已有行；合并时 **保留 USER 自由文本**（非 `key: value` 行）
- KV 行须使用 **`key: value`（冒号+空格）**；含冒号但无空格的叙述行不会被当作 KV
- L1 USER/MEMORY 写入同 key 时使用 upsert
- `confirm-pending` 仅 flush HITL pending，**不**触发 L4 合并
- 每次合并写入 L2 `compliance_audit_log`（`resource_type=external_fact`, `action=merge`）

## CLI

```bash
python document/query_memory.py resolve-entity --tenant tenant1 --user user1 --mention 张三
python document/query_memory.py profile-facts --tenant tenant1 --user user1
python document/query_memory.py profile-get --tenant tenant1 --user user1
python document/query_memory.py profile-set --tenant tenant1 --user user1 \
  --facts '[{"key":"部门","value":"产品部","source":"ldap"}]'
python document/query_memory.py profile-import --tenant tenant1 --user user1 --file path/to/profile.yaml
python document/query_memory.py list-profile-users --tenant tenant1
python document/query_memory.py finalize-session --tenant tenant1 --user user1 --session sess1
python document/query_memory.py purge-tenant-l4 --tenant tenant1
python document/query_memory.py purge-user --tenant tenant1 --user user1
```

## 合规

- `purge-user`：删除外部 YAML、清空 L1 USER 文件、可选删除 `external_fact` 审计
- `purge-tenant-l4`：删除租户全部外部画像，并从 USER 移除对应 keys

## HTTP 同步（运维）

```bash
# HTTP → 本地 YAML 镜像
python scripts/l4_profile_sync_cron.py --tenant tenant1 --direction pull

# 本地 YAML → HTTP
python scripts/l4_profile_sync_cron.py --tenant tenant1 --direction push
```

## Memory 工具

- `resolve_entity(mention)` — 实体别名解析
- `fetch_profile_facts()` — 读取当前用户 L4 facts（不触发 L1 合并）

## 配置（`config/memory.yml`）

```yaml
external_profiles_dir: data/external_profiles
external_profiles_backend: file   # file | http | noop
external_profiles_http_url: null  # backend=http 时必填
external_profiles_http_timeout: 10
external_profile_cache_ttl: 300   # 秒，0=禁用缓存
external_profile_cache_backend: redis  # redis | memory（测试用 memory）
external_merge_on_finalize: true
purge_delete_external_audit: true
purge_tenant_l4_strip_user_keys: true
```

## 适配器

| 适配器 | 用途 |
|--------|------|
| `FileExternalMemoryAdapter` | 开发/单节点：YAML 文件 |
| `HttpExternalMemoryAdapter` | 生产：REST 网关（CRM/LDAP 代理） |
| `CachedExternalMemoryAdapter` | TTL 装饰器，包装上述适配器 |
| `NoOpExternalMemoryAdapter` | `backend=noop` 占位 |

HTTP API 约定（可自定义网关实现）：

- `GET /tenants/{tenant}/users` → `{"users": ["u1", ...]}`
- `GET/PUT/DELETE /tenants/{tenant}/users/{user}/profile` → YAML 等价 JSON
