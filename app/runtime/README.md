# 工作流执行引擎（预留）

LangGraph 等引擎封装在 `runtime/adapters/langgraph/`，对上层暴露 `WorkflowRuntime` Port。

Agent 层不得直接依赖 LangGraph 类型。
