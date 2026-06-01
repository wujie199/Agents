# Phase 1: 核心契约与组合根

本目录实现了基础能力层的核心契约与组合根，包含：

## 目录结构

```
domain/              # 领域 DTO
  ├── context.py     # RequestContext, ACL
  ├── evidence.py    # EvidenceBundle, Evidence
  └── task.py        # AgentTask

ports/               # Port Protocol 定义
  ├── config.py      # ConfigPort
  ├── secret.py      # SecretPort
  ├── privacy.py     # PrivacyPort
  ├── identity.py    # IdentityPort
  ├── policy.py      # PolicyPort
  ├── observability.py
  ├── model.py       # ModelPort
  ├── rag.py         # RAGPort
  ├── memory.py      # MemoryPort
  ├── tools.py       # ToolPort
  └── storage/
      ├── cache.py   # CachePort
      └── vector.py  # VectorPort

composition/         # 组合根
  ├── run_context.py # RunContext
  └── factory.py     # Fake Port 实现与构建函数
```

## 使用示例

```python
from domain import RequestContext, ACL
from composition import build_test_context, build_run_context

# 测试场景：使用 Fake Port
ctx = build_test_context()
assert ctx.policy.get_max_parallel_sends("tenant1") == 5

# 生产场景：注入真实 Port
request = RequestContext(
    tenant_id="tenant1",
    user_id="user1",
    session_id="session1",
    trace_id="trace1",
    channel="web"
)
ctx = build_run_context(
    request=request,
    rag=real_rag_port,
    memory=real_memory_port,
    tools=real_tool_port
)

# 获取模型
llm = ctx.get_model("main_llm")

# 使用 RAG
evidence = await ctx.require_rag().route_and_retrieve("query", request)
```

## 设计原则

1. **依赖倒置**：所有 Port 为 Protocol，实现可替换
2. **禁止跨层依赖**：domain/ 不依赖任何框架
3. **可测试性**：Fake Port 支持单元测试
4. **唯一入口**：模型通过 `ctx.get_model(role)` 获取

## 验收标准

- [x] Agent 单元测试可注入 Fake RAG/Tool
- [x] domain/ 无框架依赖
- [x] ports/ 为纯 Protocol 定义
- [x] RunContext 支持按需注入 Port
