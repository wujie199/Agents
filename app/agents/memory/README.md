# app/agents/memory/ — L6 Agent 记忆运行时

本目录是 **L6 Agent 层**的记忆运行时编排，职责：

- **冲突检测**：写入前检测记忆冲突（`conflict_detector.py`）
- **名字记忆**：从自我介绍中提取姓名（`name_remember.py`）
- **会话结束**：finalize 前整理 L1 记忆（`session_finalize.py`）
- **记忆工具**：Agent 可调用的 memory 工具注册（`memory_tools.py`）
- **企业记忆视图**：列表、搜索、L4 画像（`enterprise_memory.py`）
- **运行时初始化**：启动时加载配置和适配器（`memory_bootstrap.py`）
- **指标与调试**：Prometheus 指标、运行时追踪（`memory_metrics.py`、`memory_runtime_debug.py`）
- **视图**：待确认的 L1 delta 列表（`memory_views.py`）

## 与 agent_platform/memory/ 的边界

| 本目录 (L6) | agent_platform/memory/ (L5) |
|---|---|
| 编排：决定何时读写、如何冲突检测 | 适配：实现 MemoryPort 契约 |
| 运行时：bootstrap、metrics、debug | 存储：hot/cold/skill/external 适配器 |
| 工具：memory_tools 供 ReAct 调用 | 注册：memory_tool_registration |

**依赖方向**：`app/agents/memory/ → agent_platform/memory/`（L6 依赖 L5，禁止反向）
