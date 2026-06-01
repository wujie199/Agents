# 业务侧 RAG 查询指南

本文说明在**向量 / 图 / 结构化库**多后端并存时，上层如何调用 `RAGPort`，且**无需**直接依赖 Chroma、Neo4j 或 ORM。

---

## 1. 统一入口（不变）

```python
bundle = await ctx.rag.route_and_retrieve(
    query="你的自然语言问题",
    context=request,  # RequestContext
    plan=None,        # 可选，见下文
)
```

- **输入**：`query` + `RequestContext`（`tenant_id`、`acl` 等）
- **输出**：`EvidenceBundle`（`evidences[]` + `plan` + 可选 `degraded_reason`）
- **禁止**：在业务里拼 SQL/Cypher，或 `import` `rag/adapters/*`、`storage/adapters/chroma`

组合根：`build_production_context` / `build_development_context` 已装配 `RAGPortAdapter`，并在 `config/rag_pipeline.yml` 中 `retrieval.enable_router: true` 时委托 `RetrievalRouter`。

---

## 2. 三种传参方式

### 方式 A：只传 query（自动路由，默认）

```python
bundle = await ctx.rag.route_and_retrieve(
    "谁负责 PRJ-12345 的上游依赖？",
    request,
)
```

`RetrievalRouter` 会：

1. `QueryClassifier` 分类（`semantic_doc` / `graph` / `relational` / `hybrid` 等）
2. `RoutingRules` 生成 `retrieval_plan`（主/辅后端、融合策略）
3. 执行 vector / sql / graph（已注入的 Port 才生效）
4. 融合后返回 `EvidenceBundle`；`bundle.plan` 记录实际计划

配置：`retrieval.auto_route: true`（默认）。

---

### 方式 B：Router Agent 下发 `plan`（推荐编排层）

Worker **不选库**，只消费 Router 输出的 `retrieval_plan`：

```python
section_plan = {
    "primary": "vector",
    "secondary": [],
    "top_k": 10,
    "rerank_top_n": 5,
    "fusion": "rrf",
    "cache_policy": "read_through",
}

bundle = await ctx.rag.route_and_retrieve(
    query=f"{section_title} {keywords}",
    context=request,
    plan=section_plan,
)
```

| 字段 | 说明 | 取值示例 |
|------|------|----------|
| `primary` | 主后端 | `vector`, `sql`, `graph` |
| `secondary` | 辅后端列表 | `["vector"]`, `["sql", "graph"]` |
| `fusion` | 融合 | `rrf`, `weighted`, `cascade`, `first_match` |
| `order` | 执行顺序 | `parallel`, `cascade` |
| `top_k` | 召回条数 | `10` |
| `rerank_top_n` | 重排保留 | `5` |
| `graph_hop` | 图遍历跳数 | `0`～`2` |
| `cache_policy` | 缓存 | `read_through`, `no_cache` |

批量写章：

```python
from core.ports.rag import RetrieveRequest

requests = [
    RetrieveRequest(
        query=f"{s.title} {s.keywords}",
        plan_override={"primary": "vector", "top_k": 8},
    )
    for s in sections
]
bundles = await ctx.rag.route_and_retrieve_batch(requests, request)
```

---

### 方式 C：强制指定后端（特例）

```python
# 仅结构化库（需 enable_sql: true 且组合根注入 RelationalPort）
bundle = await ctx.rag.route_and_retrieve(
    "项目 PRJ-12345 审批状态",
    request,
    plan={"primary": "sql", "secondary": [], "top_k": 1, "fusion": "first_match"},
)

# 图 + 向量（需 enable_graph: true 且注入 GraphPort）
bundle = await ctx.rag.route_and_retrieve(
    "谁负责 PRJ-12345",
    request,
    plan={
        "primary": "graph",
        "secondary": ["vector"],
        "graph_hop": 2,
        "fusion": "cascade",
    },
)
```

仍只传**自然语言** `query`；SQL/图查询在 RAG 内部参数化执行。

---

## 3. 关闭自动路由（固定向量）

```yaml
# config/rag_pipeline.yml
retrieval:
  enable_router: true
  auto_route: false
  primary_backend: vector
```

此时未传 `plan` 时，始终走向量主路径。

---

## 4. 完全关闭 Router（仅向量实现）

```yaml
retrieval:
  enable_router: false
```

`RAGPortAdapter` 仅执行内置向量检索（与 MVP 最初行为一致）。

---

## 5. 消费结果

```python
for ev in bundle.evidences:
    if ev.source_type.value == "vector":
        ...
    elif ev.source_type.value == "sql":
        ...
    elif ev.source_type.value == "graph":
        ...

if bundle.is_degraded():
    # 空结果或部分失败，勿当「无资料」
    ...
```

---

## 6. 配置与组合根

| 配置项 | 含义 |
|--------|------|
| `enable_router` | `RAGPort` 是否委托 `RetrievalRouter` |
| `auto_route` | 无 `plan` 时是否自动分类 |
| `enable_graph` | 是否允许图后端（还需 `graph_port`） |
| `enable_sql` | 是否允许 SQL 后端（还需 `sql_port`） |
| `enable_graph_index` | 加载时是否建图（二期） |

`build_rag_stack(..., sql_port=relational, graph_port=graph)` 仅在对应 `enable_*` 为 true 时传给 Router。

---

## 7. 相关代码

| 模块 | 路径 |
|------|------|
| Port 契约 | `ports/rag.py` |
| 门面 | `rag/adapters/rag_port_adapter.py` |
| 路由器 | `rag/router/router.py` |
| plan 转换 | `rag/retrieval/plan_codec.py` |
| 装配 | `composition/rag_factory_helpers.py` |
