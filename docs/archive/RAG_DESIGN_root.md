# RAG 子系统设计 — 多存储与企业级路由

> **文档体系（四份之一）**  
> **接口契约**（主文档）：`ARCHITECTURE.md` **§7.2、§10.3**；`ports/rag.py`、`domain/evidence.py`  
> **主文档**：**1.3.1 MVP**、**1.5 ADR**、协作 §9、平台模型 §13  
> 其它：`MEMORY_DESIGN.md`、`TOOLS_SKILLS_MCP_DESIGN.md`  
> **本文档**：RAG **写入 + 检索** 全部实现细节（**非**全栈独立「摄取层」）

---

## 0. 与主文档的分工

| 主题 | 主文档 `ARCHITECTURE.md` | 本文档 |
|------|--------------------------|--------|
| 五层模型 | 领域能力 + 平台 L3/L1；**无**全局 L2 摄取层 | `rag/` 内写入与读取 |
| **RAGPort / IngestPort** | 方法签名、`EvidenceBundle` 字段 | Adapter、router、Redis、ingest 流水线 |
| **L3 Port** | 抽象能力表（§5） | RAG 如何用 Cache/Vector/Graph（§3、§4） |
| 脱敏 | PrivacyPort §11 | 向量库、qc、ingest（§16） |
| 模型 | `get_model(role)` §13 | embedding/rerank/index_version §16 |

> **阅读顺序**：§0 → §1 → §2（写入）→ §3（L3 用法）→ §4 起（存储/路由/Redis/Port）。

---

## 1. RAG 在分层中的位置

- 属于**领域能力**中的 **`rag/` 子系统**；经 **Port** 使用平台 L3、L1，不直连 SDK。
- Agent **禁止**直接访问 Chroma / Redis / SQL / 图库；只能 `route_and_retrieve` / `route_and_retrieve_batch`。
- 生成（LLM）在 Agent；RAG 只返回 `EvidenceBundle`（主文档 **§10.3**、**ADR-3**）。
- **Embedding、Rerank** 经 `get_model("embedding"|"rerank")`；禁止在 Adapter 内读 `rag.yml` 自建 Chat 客户端。

### 1.1 代码模块（`rag/`）

| 路径 | 职责 |
|------|------|
| `rag/router/` | 规则 + 分类器 + 融合；消费 `retrieval_plan`（**ADR-7**） |
| `rag/adapters/` | vector_store、redis_cache、graph/sql retriever |
| `rag/ingest/`、`rag/index/` | **写入管道**（§2） |
| `rag/chains/` | 与业务生成链衔接（如 `search_create_basic_rag`） |
| `rag/eval/` | RAGAS（§16.4） |

**配置**：`config/rag.yml`、`chroma.yml`、`rag_router.yml`（Adapter 读取；编排节点不得传入 `chroma_k` 等）。

---

## 2. 摄取与索引（RAG 写入管道）

**归属 RAG 子系统**（`rag/ingest`、`rag/index`），与 **`RAGPort` 只读检索** 分离，避免「检索服务里顺便入库」。

### 2.1 IngestPort（`rag/ingest/`）

**输入**：Word/PDF/图片/HTML 等。

**流程要点**

1. 可选 `yolo` 版面检测 → 区域裁剪
2. `ocr` 或原生文本层提取（`get_model("ocr")` / `get_model("yolo")`，见 §14）
3. 规范化文本块 + metadata（`doc_id`, `tenant_id`, `acl`, `source_path`）
4. 可复用 `read_word_2_json` **实现逻辑**（`rag/ingest/adapters/`），经 **IngestPort** 调用；**ADR-2**：不暴露为 Agent 默认可调 Tool

**失败降级**：YOLO 失败 → 整页 OCR；OCR 失败 → 标记 `ocr_status=skipped`，有文本层则继续。

**合规**：入库前 `PrivacyPort.classify_sensitivity` + 可选 PII 扫描（§16）。

### 2.2 IndexPort（`rag/index/`）

1. 分块（递归/语义/Markdown 结构，可配置）
2. `get_model("embedding")` 向量化
3. 经 **VectorPort**（平台 L3）写入；更新 **`index_version`**
4. 可选写 Relational **outbox**，异步刷新 Redis `meta`（§4）

**硬约束**：更换 embedding 模型必须 bump `index_version`（§14.2）。

### 2.3 与 RAGPort 的分工

- **写入**：本节的 Ingest / Index；由编排层 `ingest_document` 或 `call_port("ingest", ...)` 触发。
- **读取**：`RAGPort` 路由到 Cache / Vector / Graph / SQL。
- 生成类 LLM 在 **Agent** 经 `main_llm` 调用，不属于写入管道。

---

## 3. 存储 Port 在 RAG 中的用法

主文档 **§5** 定义 L3 抽象能力；本节约定 **RAG 域**如何调用（实现仍在 `storage/adapters/`）。

### 3.1 CachePort（Redis）

RAG 使用的 category（key 格式 `{tenant}:{env}:rag:{category}:{id}`，详见 §4）：

- `qc` — 查询结果缓存（`EvidenceBundle` 序列化）
- `emb` — Embedding 缓存（key 含 `embedding_model_version`）
- `sess` — 多轮检索上下文摘要（**Evidence 摘要**，非对话全文；对话 sess 见 `MEMORY_DESIGN.md` §9）
- `lock` — 索引重建分布式锁
- `rate` — RAG QPS（可与 PolicyPort 共用计数）
- `meta` — `index_version`、`acl_version`，检索前比对以防脏读

### 3.2 VectorPort

- 向量写入、相似检索、按 `doc_id` 删除。
- 按 `collection` + **`index_version`** 隔离；Chroma/Milvus 在 `storage/adapters/chroma/`。
- 编排 / Agent **不感知** collection 名；由 Adapter 读配置。

### 3.3 GraphPort

- 实体链接、k-hop 子图、路径导出为文本供融合层使用。
- **检索策略**在 `rag/router/`；图存储经 GraphPort Adapter（三期，见主文档 1.3.1）。

### 3.4 RelationalPort（RAG 相关）

- **outbox**：索引异步更新（与向量库最终一致）。
- **只读 SQL 路径**（二期）：NL→`QuerySpec`→参数化查询，结果映射为 `EvidenceBundle`（§8）。

---

## 4. 四类存储的职责边界

### A. Redis（热路径 / 会话态 / 缓存）

- **不是**权威知识库，而是低延迟、可失效的数据面。
- 用途：查询结果缓存、Embedding 缓存、会话检索上下文、索引版本元数据、分布式锁、限流。
- 详见第 4 节（Redis 键设计）。

### B. 结构化数据库（PostgreSQL / MySQL 等）

- 权威业务数据：用户、订单、权限、配置、指标、主键精确查询。
- 支持事务、审计字段、软删除、行级安全（RLS）或应用层 `tenant_id` 强制过滤。

### C. 传统 RAG（向量 + 可选 BM25）

- 非结构化 / 半结构化文档：手册、规范、历史报告段落。
- 对齐现有 `VectorStoreService`、`RagSummarizeService`、`build_search_create_rag_query` 场景。

### D. Graph RAG（知识图谱 + 图遍历）

- 实体—关系—属性、多跳推理、影响链、依赖链、组织关系。
- 与向量协同：Graph 扩展子图 → 文本化路径叙述 → 进入融合层。

---

## 5. 检索路由器（Retrieval Router）

### 5.1 输入

- `query`：自然语言或结构化查询意图
- `query_type`（分类器输出）：`factual_exact` | `semantic_doc` | `relational` | `hybrid` | `operational`
- `entities`：NER 或规则抽取的实体列表
- `session_context`：来自 Memory 冷档案 / Redis `sess` 的近期摘要
- `acl`：当前用户可访问的文档集合、表、图命名空间

### 5.2 决策流程

```
Step 0: 读 Redis 查询缓存
  Key: query_hash + tenant_id + acl_version
  → 命中且未过期 → 直接返回 EvidenceBundle（可跳过 Step 1–4）

Step 1: 规则优先（确定性，可审计）
  - 含明确主键/编号/表名/「等于」「统计」「列表」→ 结构化 DB 为主
  - 含「谁负责」「依赖」「影响」「关联」「上下游」→ Graph RAG 为主
  - 长文档语义复述、规范条款、案例描述 → 传统 RAG 为主

Step 2: 分类器输出 retrieval_plan
  - backends: 可选 redis_cache, sql, vector, graph
  - order: parallel 或 cascade
  - fusion: rrf | weighted | cascade

Step 3: 执行计划
  - SQL: 仅参数化查询，禁止拼接用户原始输入
  - Vector: embedding → top-k → rerank（对齐 config/rag.yml、chroma.yml）
  - Graph: 实体链接 → k-hop 子图 → 路径摘要

Step 4: 融合与裁剪
  - 去重：doc_id / chunk_hash / entity_id
  - 重排：cross-encoder 或 rerank API
  - 输出 EvidenceBundle，标注 source_type

Step 5: 写回 Redis
  - 完整 EvidenceBundle 短期缓存（qc）
  - 可选 embedding 中间缓存（emb）
```

### 5.3 何时仅使用单一后端

**仅 Redis**

- 完全相同 query 在 TTL 内重复请求
- 会话内刚检索过的片段（配合 `sess` 环形列表）
- Embedding 向量已缓存且 `model_version` 未变

**仅结构化 DB**

- 精确 ID、状态机查询、聚合报表、权限表、租户配置
- 时间范围 + 数值过滤且 schema 已知
- 强一致性要求的实时指标

**仅传统 RAG**

- 文档型问答、无图谱实体、相似段落召回即可
- 工程报告章节撰写：章节名 + 子标题 + 原文 excerpt 构造 query（现有模式）

**仅 Graph RAG**

- 关系路径问题、根因分析、供应链 / 组织关系
- 实体已在图中且链接置信度高于阈值

**组合（企业推荐默认：hybrid）**

- SQL 过滤权限与业务范围 → 向量召回 → Graph 扩展关联实体 → 融合
- operational：先 Redis 会话上下文 → SQL 取实时状态 → 必要时补向量背景

### 5.4 与 Router Agent 的衔接

`Router.py` 输出除章节/工具映射外，扩展 `retrieval_plan` 字段示例：

```json
{
  "route_id": "search_create",
  "retrieval_plan": {
    "primary": "vector",
    "secondary": [],
    "graph_hop": 0,
    "cache_policy": "read_through"
  }
}
```

Worker 节点只消费 `retrieval_plan`，不自行选择存储后端。

---

## 6. Redis 企业级设计

### 6.1 Key 命名规范

格式：`{tenant}:{env}:rag:{category}:{id}`

**category 说明**

- `qc` — query cache，完整检索结果
- `emb` — embedding 缓存，text_hash → vector
- `sess` — 会话检索上下文，最近 N 次 Evidence 摘要
- `lock` — 分布式锁，索引重建、批量失效
- `rate` — 限流计数
- `meta` — 索引版本、acl 版本号

### 6.2 各 category 存什么

**qc（查询缓存）**

- Value：序列化 `EvidenceBundle`（JSON 或 MessagePack，可选压缩）
- TTL：5–30 分钟（可配置）；含 PII 或高敏内容时缩短 TTL 或禁用 qc
- 失效：文档版本变更、acl 版本 bump、主动 `invalidate_pattern`

**emb（Embedding 缓存）**

- Key：`emb:{model_version}:{text_hash}`
- Value：float 数组或 base64
- TTL：7–30 天；模型升级时 bump `model_version` 前缀即批量失效

**sess（会话检索态）**

- Key：`sess:{session_id}:recent_evidence`
- Value：环形列表，元素为 `{query_hash, summary, source_ids, ts}`
- TTL：与会话一致，默认 24h 滑动续期
- 用途：多轮指代（「上一段提到的规范」）

**lock**

- 索引重建、全量 re-embed 互斥；带超时自动释放，防死锁

**rate**

- 按 `user_id` / `tenant_id` 限制 RAG QPS 与 token 预算

**meta**

- `index_version:{collection}`、`acl_version:{tenant}`
- 检索前比对：版本不一致则跳过 qc，避免脏读

### 6.3 更新策略

- **Cache-aside**：读 miss → 查 DB/向量/图 → 填 qc
- **Write-through（异步）**：检索成功后后台写 qc，不阻塞主路径
- **禁止**将 Redis 作为唯一真相源；权威数据在 SQL / 向量库 / 图库

### 6.4 删除与失效

**单文档更新**

1. 向量库按 `doc_id` 删除旧 chunk 再写入新 chunk
2. 维护反向索引 `doc:{doc_id} -> Set[query_hash]`，删除关联 qc 条目
3. bump `meta:index_version:{collection}`

**租户 ACL 变更**

- bump `meta:acl_version:{tenant}`，该租户 qc 全部 miss

**用户数据删除（合规）**

- 删 SQL 记录 + 向量 chunk + 图节点 + 失效 qc/sess + Memory 冷档案 anonymize

**全量重建**

- 获取 `lock` → 新 collection 双写 → 切换 `index_version` → 异步清理旧 qc

### 6.5 一致性

- Redis 与向量库：**最终一致**；用 `index_version` 降低 stale 风险
- SQL 与向量：**Outbox 模式** — 事务写 outbox 表，worker 消费后更新向量/图

---

## 7. 传统 RAG 流水线

### 7.1 摄取

- Loader → 清洗 → 分块（递归 / 语义 / Markdown 结构）
- Metadata 必填：`tenant_id`, `doc_id`, `chunk_id`, `acl`, `content_hash`, `indexed_at`
- 对齐现有 `md5` 去重：相同 hash 跳过重复索引

### 7.2 索引

- Embedding（如 `bge-m3`）→ Chroma / Milvus
- 可选 BM25 并行索引，混合检索权重来自 `chroma.yml`

### 7.3 检索

- `k`、混合权重、`rerank_top_n` 来自配置文件
- 查询改写（HyDE、多查询）仅在高价值场景通过 feature flag 开启

### 7.4 与生成层边界

- `assemble_search_create_new_project_data` 类逻辑留在领域层：将 `EvidenceBundle` 格式化为 prompt 变量
- RAG 层不调用 LLM（评估链路除外）

---

## 8. Graph RAG 流水线

### 8.1 构图

- 实体抽取（LLM + 规则）→ 消歧 → 写入图数据库
- 边类型可配置：`REFERENCES`, `DEPENDS_ON`, `OWNED_BY`, `LOCATED_IN` 等

### 8.2 检索

- 实体链接 → 参数化图查询模板（Cypher / Gremlin）→ k-hop 上限（防子图爆炸）
- 子图线性化为「路径叙述」送入融合层

### 8.3 与向量协同（GraphRAG 路径）

- 向量 top-k 文档 → 抽取实体 → 图扩展 → 与向量结果融合

---

## 9. 结构化查询层

- **Query Builder**：自然语言 → `QuerySpec`（表、字段、过滤）→ 规则校验 → 参数化执行
- 使用**只读**数据库账号；敏感表禁止通过 NL2SQL 自动暴露
- 结果映射为 `EvidenceBundle`，`source_type=sql`

---

## 10. RAGPort 接口契约

```python
class RAGPort(Protocol):
    async def route_and_retrieve(
        self,
        query: str,
        context: RequestContext,
        plan: RetrievalPlan | None = None,
    ) -> EvidenceBundle: ...

    async def route_and_retrieve_batch(
        self,
        requests: list[RetrieveRequest],
        context: RequestContext,
        plan: RetrievalPlan | None = None,
    ) -> list[EvidenceBundle]: ...

    async def invalidate_document(self, doc_id: str, tenant_id: str) -> None: ...

    async def health(self) -> dict: ...
```

**RetrieveRequest**（`domain/` 建议字段）

- `query: str`
- `context_key: str | None` — 章节/段落关联键，便于审计
- `plan_override: RetrievalPlan | None` — 单条覆盖，缺省用 batch 级 `plan`

**批处理语义**（对齐主文档 9.7 节 B 类）

- Router **一次**生成或合并 `RetrievalPlan`（或按 query 分组计划），避免 N 次重复规划。
- 多路召回在 Adapter 内 **Semaphore 限流** 并行，上限读 `config/concurrency.yml` 的 `max_rag_batch_queries`。
- Embedding / Rerank 走 **批量** `get_model("embedding")`，`batch_size` 受 `max_embed_batch_size` 约束。
- Agent / 编排 **禁止**循环单条 `route_and_retrieve`；应调用 `route_and_retrieve_batch` 或由编排多次 `Send`（A 类）拆分。

**与现有实现**

- `rag/rag_service.py` 中 `retrieve_batched_from_context_list` 收敛为 `RAGPort` 的 batch 方法之一。

**RetrievalPlan 字段建议**

- `primary_backend`: vector | sql | graph | redis_only
- `secondary_backends`: list
- `fusion_strategy`: str
- `cache_policy`: none | read_through | write_through
- `max_hop`: int（Graph）
- `top_k`, `rerank_top_n`: int

---

## 11. EvidenceBundle 结构

每条 evidence 项包含：

- `id`：来源标识（chunk_id / row_id / entity_id）
- `content`：文本或序列化片段
- `source_type`：vector | sql | graph | cache
- `score`：相关性分数
- `citation`：文件路径、表名、图路径等引用信息
- `metadata`：acl、版本、时间戳

整体字段见主文档 **§10.3**（`empty`、`degraded_reason`、`error_code` 等）。

---

## 12. 配置建议（rag_router.yml）

```yaml
router:
  default_plan:
    primary: vector
    secondary: []
    fusion: rrf
    cache_policy: read_through
  rules:
    - pattern: "(统计|列表|等于|编号)"
      plan: { primary: sql }
    - pattern: "(依赖|关联|影响|上下游)"
      plan: { primary: graph, secondary: [vector] }
redis:
  qc_ttl_seconds: 900
  emb_ttl_seconds: 604800
  sess_max_items: 20
```

---

## 13. 评估与治理

- **离线**：RAGAS（`rag/run_ragas.py`）、检索命中率、延迟 P95
- **在线**：用户反馈 → 反馈表 → 调整 router 规则权重
- **审计**：记录 `plan`, `backends_hit`, `doc_ids`；日志不写入完整 PII 正文

---

## 14. 降级顺序

当某一后端失败或超时时：

1. Graph 失败 → 降级向量 + 关键词
2. 向量失败 → 仅 SQL（若 query 含结构化意图）或返回空 Evidence + 明确错误码
3. 全部失败 → `EvidenceBundle.empty=true` + `degraded_reason` + `error_code`（**ADR-3**）；报告场景可回退 ACL 内原文 excerpt，且 `observation` 须 `degraded=true`

---

## 15. Phase 落地顺序

> **一期范围以 `ARCHITECTURE.md` 1.3.1 为准**；下列步骤 3～5 为二期及以后。

1. 封装现有 Chroma 检索为 `RAGPort` 向量适配器 + 可选 Redis `emb`（`qc` 可后置）
2. Router 输出 `retrieval_plan`（一期可固定 `primary: vector`）；`route_and_retrieve_batch` 对齐 `retrieve_batched`
3. （二期）规则 + 配置驱动的 `RetrievalRouter` 全量；只读 SQL `QuerySpec`
4. （三期）Graph 构图与 k-hop 检索
5. 全链路 RAGAS 与失效自动化测试（`index_version` 迁移纳入 runbook）

---

## 16. 模型协作（L1 ModelRegistry）

配置与主 LLM 降级见 `ARCHITECTURE.md` 第 13 节；脱敏见第 11 节；本节仅写 RAG 域内规则。

### 16.1 实例获取

- 向量入库与检索 → `get_model("embedding")`
- 检索后重排 → `get_model("rerank")`
- 基于 Evidence 的段落生成 → `get_model("main_llm")`（生成降级与检索降级相互独立）

### 16.2 Embedding 降级与索引版本

- **同 profile 重试**：网络/502/503/504 可重试，不切换模型 id。
- **禁止**：在未 bump 索引版本时，用不同 embedding profile 写入同一 Chroma collection。
- **允许迁移**：新 collection 全量 re-embed → 切换读路径 → 废弃旧 collection；`meta:index_version:{collection}` 随迁移 bump。
- Redis `emb` 缓存 key 必须含 **`embedding_model_version`**（与 `models.yml` embedding profile 联动）。

### 16.3 Rerank 降级

- 失败时：**跳过重排**，使用向量（+ BM25）融合后的 top-k 直接进入生成。
- 可选备用 rerank profile：须 A/B 验证分数分布与主 profile 接近。
- `rerank_top_n`、`k`、混合权重仍在 `chroma.yml` / `rag_router.yml`；模型构造在 `models.yml` 的 `roles.rerank`。

### 16.4 评估（RAGAS）

- RAGAS 的 LLM / Embeddings 须显式使用 `main_llm`、`embedding` role，与生产一致，避免测评绕过线上降级链。

---

## 17. 合规、脱敏与缓存（ADR-4）

与主文档 **PrivacyPort（§11）** 配合；本节仅 RAG 域。

### 17.1 向量库与入库

- **默认**：合同/规范类文档 **全文入库 + ACL**（检索质量优先）。
- **高敏租户**：`privacy.yml` 开启入库前 `redact_for_storage`（质量下降须在产品中说明）。
- **metadata**：避免存身份证号等明文；强制 `tenant_id` + `acl` 隔离。
- **ingest**：`classify_sensitivity` + 可选 PII 扫描后再写入 VectorPort（§2.1）。

### 17.2 Redis qc / emb / sess

- **qc**：含用户原文或 PII 的 query → **缩短 TTL 或禁用 qc**；`acl_version` bump 后全部 miss。
- **emb**：key 含 `embedding_model_version`；模型升级时前缀失效（§16.2）。
- **sess**：只存 **Evidence 摘要**，不存完整对话。

### 17.3 出域与 Evidence 进 prompt

- 企业文档类 `EvidenceBundle` 通常 **保留原文**，由 ACL 保证不越权；对话中的手机号等可走 `redact_for_llm`（主文档 §11.1）。
- 审计：记 `plan`、`backends_hit`、`doc_ids`；**不**记完整 chunk 正文。

### 17.4 遗忘权与文档删除

与 `MEMORY_DESIGN.md` 用户删除流程联动：

1. 向量库按 `doc_id` 删除 chunk  
2. 失效关联 **qc**（二期：`doc_id → query_hash` 反向索引）及 `meta:index_version`  
3. 失效 Redis `sess` 中相关摘要  

---

## 18. 与现有代码映射（演进）

| 现状（`My_Agent_Project/`） | 目标落点 |
|-----------------------------|----------|
| `rag/vector_store.py` | `storage/adapters/chroma/` + `rag/adapters/` |
| `rag/rag_service.retrieve_batched_*` | `RAGPort.route_and_retrieve_batch` |
| `rag/search_create_basic_rag.py` | `rag/chains/` |
| `tools/read_word_2_json` 逻辑 | `rag/ingest/adapters/`（经 IngestPort，ADR-2） |
| `Agents/MainAgent` 内联 RAG 调用 | Agent Worker + `ctx.rag` |
