# 工作流执行引擎

LangGraph 封装在 `runtime/adapters/langgraph/`，聊天场景在 `workflows/chat/`。

## 聊天 REPL（LangGraph，默认）

```bash
python app/chat_repl.py --tenant tenant1 --user user1 --session chat1
python app/chat_repl.py --engine direct    # 直连 run_chat_turn（含 Path B ReAct）
python app/chat_repl.py --no-rag
python app/chat_repl.py --no-tools         # Path A，无记忆工具
python app/chat_repl.py --stream           # 流式打印 assistant
python app/chat_repl.py --profile production
python app/chat_repl.py --debug
```

REPL 命令：`/snapshot` L1、`/turns` L2、`/pending` 待确认记忆、`/confirm` 写入 L1。

图拓扑（Path B 默认，可用 `--no-tools` 关闭）：

```
prepare (L1 + Context + RAG) → agent (session_search, remember_user_fact) → persist
```

`thread_id` = `session_id`，Checkpointer 默认 `MemorySaver`。

HTTP API 见 `app/api/README.md`。

Agent 层不直接 `import langgraph`；业务入口为 `app/agents/chat_langgraph.py`。
