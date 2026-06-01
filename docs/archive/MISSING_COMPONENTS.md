# 基础能力层完整性检查

## 一、L1 基础设施层

| Port | 定义 | 实现 | 状态 |
|------|------|------|------|
| ConfigPort | ports/config.py | infrastructure/config/adapter.py | ✅ 已实现 |
| SecretPort | ports/secret.py | infrastructure/secret/adapter.py | ✅ 已实现 |
| PrivacyPort | ports/privacy.py | infrastructure/privacy/adapter.py | ✅ 已实现 |
| ObservabilityPort | ports/observability.py | infrastructure/observability/adapter.py | ✅ 已实现 |
| IdentityPort | ports/identity.py | infrastructure/identity/adapter.py | ✅ 已实现 |
| PolicyPort | ports/policy.py | infrastructure/policy/adapter.py | ✅ 已实现 |
| **ModelPort** | ports/model.py | ❌ 缺失 | ⚠️ 需实现 |

**缺失**：ModelPort Registry（模型注册中心）

---

## 二、L3 存储层

| Port | 定义 | 实现 | 状态 |
|------|------|------|------|
| CachePort | ports/storage/cache.py | storage/adapters/redis/cache_adapter.py | ✅ 已实现（企业级） |
| VectorPort | ports/storage/vector.py | storage/adapters/chroma/vector_adapter.py | ✅ 已实现 |
| RelationalPort | ports/storage/relational.py | storage/adapters/sqlite/relational_adapter.py | ✅ 已实现（企业级） |
| GraphPort | ports/storage/graph.py | storage/adapters/neo4j/neo4j_graph_adapter.py | ✅ 已实现（企业级） |
| ObjectStorePort | ports/storage/object_store.py | storage/adapters/s3/s3_object_store_adapter.py | ✅ 已实现（企业级） |

**完整度**：5/5 (100%)

---

## 三、L5 领域能力层

| Port | 定义 | 实现 | 状态 |
|------|------|------|------|
| **RAGPort** | ports/rag.py | ❌ 缺失 | ⚠️ 需实现 |
| **MemoryPort** | ports/memory.py | ❌ 缺失 | ⚠️ 需实现 |
| **ToolPort** | ports/tools.py | ❌ 缺失 | ⚠️ 需实现 |
| SkillPort | ports/skills.py | infrastructure/skills/adapter.py | ✅ 已实现 |
| MCPPort | ports/mcp.py | infrastructure/mcp/adapter.py | ✅ 已实现（企业级） |
| ModelPort | ports/model.py | ❌ 缺失 | ⚠️ 需实现 |

**完整度**：2/6 (33%)

---

## 四、缺失项详细分析

### 1. ModelPort Registry（P0 必需）

**需求**：
- 统一 `get_model(role)` 入口
- 模型降级链（main_llm → router_llm → 本地）
- 熔断器
- 缓存管理
- 配置驱动（config/models.yml）

**影响范围**：
- Agent 调用 LLM
- RAG 使用 embedding/rerank
- 所有生成类任务

**文件结构**：
```
model/
├── registry.py          # ModelRegistry 主类
├── factory.py           # get_model 入口
├── providers/
│   ├── dashscope.py     # 阿里云
│   ├── openai.py        # OpenAI
│   └── local.py         # 本地模型
└── resilience/
    ├── circuit_breaker.py
    └── retry.py
```

---

### 2. RAGPort Adapter（P0 必需）

**需求**：
- 封装现有 `rag/` 目录代码
- 实现 `route_and_retrieve` / `route_and_retrieve_batch`
- 路由逻辑（vector/sql/graph）
- 降级策略（ADR-3）
- 与 VectorPort、CachePort 集成

**影响范围**：
- 所有检索类任务
- 报告生成章节内容

**文件结构**：
```
rag/
├── adapters/
│   └── rag_port_adapter.py  # RAGPort 实现
├── router/
│   └── router.py             # 路由逻辑
└── ingest/
    └── ingest_port.py        # 写入管道
```

---

### 3. MemoryPort Adapter（P1 重要）

**需求**：
- Hermes L1 热记忆（MEMORY + USER）
- L2 冷档案（Session Archive）
- L3 技能记忆
- `compose_prompt_snapshot`
- `persist_turn`
- `session_search`

**影响范围**：
- 会话上下文
- 用户偏好
- 跨轮对话

**文件结构**：
```
memory/
├── adapters/
│   └── memory_port_adapter.py
├── hot/
│   └── prompt_memory.py      # L1 热记忆
├── archive/
│   └── session_archive.py    # L2 冷档案
└── skills/
    └── skill_memory.py       # L3 技能记忆
```

---

### 4. ToolPort Adapter（P1 重要）

**需求**：
- 包装现有 `tools/` 目录函数
- ACL 校验
- 审计记录
- 幂等性支持
- 配置驱动（config/tools.yml）

**影响范围**：
- 所有工具调用
- 文件读写
- 结果持久化

**文件结构**：
```
tools/
├── adapters/
│   └── tool_port_adapter.py
├── registry.py              # 工具注册表
└── (现有工具函数)
```

---

## 五、优先级排序

| 优先级 | Port | 原因 | 阻塞项 |
|--------|------|------|--------|
| P0 | ModelPort Registry | Agent 无法调用 LLM | 所有生成任务 |
| P0 | RAGPort Adapter | 无法检索 | 报告生成核心功能 |
| P1 | ToolPort Adapter | 无法调工具 | 文件操作、结果保存 |
| P1 | MemoryPort Adapter | 无会话记忆 | 多轮对话体验差 |

---

## 六、总结

**当前完成度**：

```
L1 基础设施:  6/7 (86%)  缺 ModelPort
L3 存储:      5/5 (100%) ✅
L5 领域:      2/6 (33%)  缺 RAG/Memory/Tool/Model

总计: 13/18 (72%)
```

**需要补充**：

1. ✅ **ModelPort Registry**（P0）- 模型注册中心
2. ✅ **RAGPort Adapter**（P0）- RAG 封装
3. ✅ **ToolPort Adapter**（P1）- 工具包装
4. ✅ **MemoryPort Adapter**（P1）- 记忆管理

**建议实施顺序**：

```
Phase 1: ModelPort Registry (1天)
  → Agent 可调用 LLM
  
Phase 2: RAGPort Adapter (1天)
  → 可检索知识库
  
Phase 3: ToolPort Adapter (0.5天)
  → 可调用工具
  
Phase 4: MemoryPort Adapter (1天)
  → 会话记忆完整
```
