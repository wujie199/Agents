# L7 应用编排（预留）

场景 DAG 定义目录。报告生成场景：`workflows/report_generation/`。

编排节点通过 `RunContext` 调用 `knowledge_base`、`rag`、`tools` 等 Port。
# workflows/report_generation/ — 报告生成场景

## 迁移状态

🚧 **迁移进行中** — 当前编排逻辑仍在 `app/agents/orchestration/`，待迁入本目录。

## 迁移计划（ARCHITECTURE.md §15.1 S5）

从 `app/agents/orchestration/` 迁出：

| 原位置 | 目标 |
|---|---|
| `chat_langgraph.py` 图编译 | `graph_def.py` — 仅拓扑与 compile 参数 |
| `chat_nodes.py` 节点 | `nodes.py` — 薄节点（<30 行），业务主路径 |
| `chat_service.py` | `service.py` — 场景服务入口 |

## 迁移后约束

- 节点函数仅含业务调用，工程化横切走 Middleware
- 禁止 `import chromadb`、`redis`、`langgraph.graph`（除 graph_def.py）
- 并行走 `Send`，禁止节点内 `ThreadPoolExecutor`
