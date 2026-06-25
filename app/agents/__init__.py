"""L6 Agent 模块。"""

from app.agents.orchestration.chat_turn import ChatTurnResult, run_chat_turn
from app.agents.orchestration.chat_langgraph import (
    ChatLangGraphSession,
    create_chat_langgraph_session,
    run_chat_turn_langgraph,
)
from app.agents.roles.react_loop import (
    end_agent_session,
    execute_tool_calls,
    run_agent_turn,
)

__all__ = [
    "ChatTurnResult",
    "ChatLangGraphSession",
    "run_chat_turn",
    "create_chat_langgraph_session",
    "run_chat_turn_langgraph",
    "run_agent_turn",
    "execute_tool_calls",
    "end_agent_session",
]

# DeepAgents 规划层（可选依赖，不装不报错）
try:
    from app.runtime.adapters.deepagents import is_deep_agents_available

    if is_deep_agents_available():
        from app.runtime.adapters.deepagents.adapter import DeepAgentAdapter
        from app.runtime.adapters.deepagents.routing_gate import (
            should_use_deep_agent,
            should_use_deep_agent_async,
        )

        __all__.extend([
            "DeepAgentAdapter",
            "should_use_deep_agent",
            "should_use_deep_agent_async",
        ])
except ImportError:
    pass
