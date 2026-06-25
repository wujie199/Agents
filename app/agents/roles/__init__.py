"""Agent 角色：Router / Worker / ReAct 执行。

通过精确路径导入，避免循环引用：
    from app.agents.roles.retrieval_router import RetrievalPlan
    from app.agents.roles.react_turn import make_react_agent
    from app.agents.roles.react_loop import run_agent_turn
"""
