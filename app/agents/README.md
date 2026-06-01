# L6 Agent 层（预留）

Router / Worker / Reviewer 角色实现将放在 `agents/roles/`。

- `router.py` — 章节路由、`retrieval_plan`
- `worker.py` — RAG 检索、内容生成
- `reviewer.py` — 质量审查

业务通过 `RunContext` 使用 Port，不直接 import 本目录实现类。
