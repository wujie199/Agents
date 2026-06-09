# L3 Skills（程序性记忆）

## 目录

| 路径 | 用途 |
|------|------|
| `skills/published/` | 已发布可执行 skill（只读） |
| `skills/drafts/{tenant}/` | 草稿 |
| `skills/meta/{tenant}/` | 运行时元数据（success_rate、anti_patterns、status） |
| `skills/source/` | CI/手工同步源（`sync-skills`） |

## CLI

```bash
python document/query_memory.py skill-search --query 报告 --tenant tenant1
python document/query_memory.py skill-list --tenant tenant1
python document/query_memory.py skill-get --skill-id json_section
python document/query_memory.py skill-run --skill-id json_section --inputs '{"title":"intro"}'
python document/query_memory.py extract-skill-draft --tenant t1 --title "X" --triggers a --steps '[...]'
python document/query_memory.py publish-skill --tenant t1 --skill-id my_skill
python document/query_memory.py deprecate-skill --skill-id example --tenant tenant1
python document/query_memory.py activate-skill --skill-id example --tenant tenant1
python document/query_memory.py sync-skills --source-dir skills/source
python document/query_memory.py skill-runs-list --tenant tenant1 --user user1
python document/query_memory.py purge-tenant-l3 --tenant tenant1
```

或：`bash scripts/sync_skills.sh [source_dir]`

## 合规说明

- `purge-user`：仅删除该用户 **skill_runs**（不删租户 drafts；draft 无 owner 字段）
- `purge-tenant-l3`：删除租户 **meta + drafts**，默认同时删除全部 **skill_runs**

## 配置（`config/memory.yml`）

```yaml
skills_meta_dir: skills/meta
skill_auto_extract_draft: false   # 成功后自动写草稿（仍须 publish）
skill_deprecate_threshold: 0.2    # 低于此 success_rate 自动 deprecated
skill_include_deprecated_in_search: false
skill_auto_extract_min_steps: 2
```

## 审计

每次 `run_skill` 在 L2 `skill_runs` 表记录执行摘要；`purge-user` 会删除该用户的 skill_runs 与租户下 drafts。

## 已发布示例

- `example` — skill_echo 演示
- `json_section` — 读/写 JSON 章节
- `session_lookup` — session_search
- `list_json_titles` — read_json_all_title
- `report_context` — skill_search + read_json 组合
