# Checkpointer 与 Session Archive

## 职责分离

| 组件 | 生命周期 | 存储内容 | 用途 |
|------|----------|----------|------|
| **Session Archive (L2)** | 长 | 完整 messages / tool_calls | 合规、检索、审计 |
| **Checkpointer** | 短 | 图状态快照 JSON | 恢复中断的 Agent/Workflow 执行 |

Checkpointer **不**替代 L2；并行 Worker 使用独立 `thread_id`（如 `{session_id}:{task_id}`），不把全量 messages merge 进父 thread。

## 实现

- Port：`core/ports/checkpointer.py`
- Adapter：`agent_platform/memory/adapters/relational_checkpointer_adapter.py`
- 表：`graph_checkpoints`（与 L2 共用 archive DB）

## 配置

`turn_buffer_flush_size`（`config/memory.yml`）> 0 时，`build_production_context` / `build_development_context` 会在 `RunContext.turn_buffer` 注入批量刷盘缓冲。

## 代码接入

```python
from app.agents.react_loop import run_agent_turn, end_agent_session

# RunContext 已含 turn_buffer / checkpointer（factory 注入）
await run_agent_turn(ctx, "hello")  # 自动使用 ctx.turn_buffer

await end_agent_session(
    ctx,
    checkpoint_state={"node": "finalize", "progress": 1.0},
)
```

## CLI

```bash
python document/query_memory.py checkpoint-health
python document/query_memory.py checkpoint-list --tenant tenant1 --limit 10
```

HTTP：`GET /health` 的 `checks.checkpointer` 字段。
