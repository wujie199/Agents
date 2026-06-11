# 租户与 RAG 索引对齐

## 记忆层（L1–L4）

| 层 | 租户键 | 说明 |
|----|--------|------|
| L1 | `tenant_id` + `user_id` | `data/memory_dev/{tenant}/USER_{user}.md` |
| L2 | `tenant_id` + `user_id` + `session_id` | SQLite/PG archive |
| L3 | `tenant_id` | `skills/meta/{tenant}/`、`skills/drafts/{tenant}/` |
| L4 | `tenant_id` + `user_id` | `data/external_profiles/{tenant}/{user}.yaml` |

Chat REPL / API 的 `--tenant` / `tenant_id` 贯穿上述各层，**无需额外配置**。

## RAG 检索租户

RAG 向量索引按 **tenant** 隔离。解析顺序（`app/agents/context_factory.resolve_rag_tenant_id`）：

1. **`RAG_TENANT_ID` 环境变量**（显式覆盖，最高优先级）
2. **production**：始终使用 `request.tenant_id`
3. **dev**：
   - 默认使用 `request.tenant_id`
   - 若存在离线 Chroma 目录 `data/rag_offline/chroma_dev`（或非空），且 **`RAG_LEGACY_DEFAULT` 未设为 false**，则回退 **`default`**，以兼容历史按 `default` 入库的索引

### 常见场景

| 场景 | 建议 |
|------|------|
| 本地 REPL，`tenant1`，沿用旧离线索引 | 默认即可（自动走 `default`） |
| 新租户索引，与 memory tenant 严格一致 | `export RAG_LEGACY_DEFAULT=false` 或 `export RAG_TENANT_ID=tenant1` |
| 生产多租户 | 按租户 re-ingest；不设 `RAG_LEGACY_DEFAULT` |

### 重建按租户索引

```bash
# 示例：为 tenant1 离线入库（见 document/rag 与 rag_pipeline.yml）
python -m document.rag.cli ingest --tenant tenant1 ...
export RAG_LEGACY_DEFAULT=false
python app/chat_repl.py --tenant tenant1 --user user1
```

## dev 开箱即用

- **记忆**：`config/memory.yml` 默认全开，启动时 `bootstrap_memory_runtime`（无需 `MEMORY_CONFIG`）
- **Chat**：`config/chat.yml` 的 `profiles.dev` 放宽 HITL（见 `ENTERPRISE_CHAT.md`）
