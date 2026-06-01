# 记忆管理子系统设计 — Hermes 四层架构

> **文档体系（四份之一）**  
> 主文档：`ARCHITECTURE.md`（**五层模型 §3**；MVP **1.3.1**；ADR **1.5**）  
> 其它：`RAG_DESIGN.md`、`TOOLS_SKILLS_MCP_DESIGN.md`  
> 本文档：`MemoryPort`、Hermes 四层记忆（**终态**；一期：热记忆快照 + 冷档案 `persist_turn`）  
> 注：Hermes **L2 = Session Archive**，与已取消的全局架构「L2 摄取」无关。  
> 参考：Hermes Agent — 稳定 Prompt 前缀 + 工具按需加载冷记忆

## 1. 设计原则

**核心约束：保持 Prompt 稳定以利于 LLM 前缀缓存；其余记忆全部通过工具按需加载。**

- L1 在会话运行期间**尽量不变**，避免 prefix cache 失效与 token 膨胀。
- L2 / L3 / L4 通过 `MemoryPort` 提供的工具（如 `session_search`、`skill_search`）拉取，不将全库灌入 prompt。
- 记忆与 RAG 知识库分离：用户偏好、会话 transcript 不进向量库（除非明确产品需求）。

---

## 2. 四层模型总览

```
┌─────────────────────────────────────────────────────────────┐
│ L1 Prompt Memory（热 / Hot）                                 │
│ MEMORY 等价 · USER 等价 · 会话内冻结                          │
│ 预算约 800 + 500 tokens · 会话结束或 HITL 后写盘              │
└─────────────────────────────────────────────────────────────┘
                              ↓ 工具按需
┌─────────────────────────────────────────────────────────────┐
│ L2 Session Archive（冷回忆 / Cold Episodic）                 │
│ 全量会话 transcript · SQLite / PostgreSQL                    │
│ 工具 session_search → LLM 二次摘要后返回                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ L3 Skills（程序性记忆 / Procedural）                         │
│ 可复用技能文档 · 步骤/工具/反模式 · skill_search              │
│ 任务成功后提取 · 审核发布 · 复用自我改进                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ L4 External Provider（可选增强）                             │
│ 实体解析 · 跨会话结构化画像 · CRM / LDAP / 工单系统            │
└─────────────────────────────────────────────────────────────┘
```

**补充：工作记忆（非 Hermes 独立层）**

- 当前任务的槽位、意图、中间结果存放在 **LangGraph `GraphState`**，任务结束清空。
- 对应原「工作记忆层」，不单独列为 Hermes 第五层，避免概念重叠。

**补充：上下文窗口（非持久层）**

- 最近 N 轮 `messages` 受模型 context 限制，由 L3 执行引擎裁剪，不写入 L2 以外的持久存储。

---

## 3. L1 — Prompt Memory（热）

### 3.1 内容拆分

**MEMORY（系统/租户级长期事实）**

- 产品规则、写作风格、组织术语表摘要、合规禁忌
- 按 `tenant_id` 命名空间隔离

**USER（用户级偏好）**

- 称呼、输出格式、常用项目、语言、禁用话题
- 按 `user_id` 命名空间隔离

### 3.2 生命周期

**会话 start**

- 从 `langgraph.store` 或文件 Store 加载 MEMORY + USER 快照
- 拼入 system prompt **固定前缀**；记录 `memory_snapshot_hash` 到 `GraphState`

**会话 running**

- 默认**不**每轮改写 L1
- 用户说「记住这个」→ 标记 `pending_memory_delta` → HITL 或高置信规则确认 → 会话 end 时合并写盘

**会话 end**

- 合并 `pending_memory_delta` 到 Store
- 超字符上限时 LLM 压缩摘要后存储

### 3.3 存储形态

- 开发：`workspace/memory/{tenant_id}/MEMORY.md`、`USER_{user_id}.md`
- 生产：LangGraph Store namespace `("memory", tenant_id, "global")` 与 `("memory", tenant_id, user_id)`

### 3.4 Token 预算

- MEMORY 上限约 2200 字符（约 800 tokens）
- USER 上限约 1375 字符（约 500 tokens）
- 超限拒绝写入或触发压缩流水线

---

## 4. L2 — Session Archive（冷）

### 4.1 存储 Schema（逻辑模型）

**sessions**

- `session_id`, `user_id`, `tenant_id`, `channel`, `started_at`, `ended_at`, `status`

**messages**

- `message_id`, `session_id`, `role`, `content`, `ts`, `token_count`, `redacted`, `metadata_json`

**tool_calls**

- `call_id`, `session_id`, `tool_name`, `args_hash`, `result_summary`, `status`, `latency_ms`

### 4.2 访问方式

**工具 `session_search`**

- 输入：`query`, 可选 `session_id`, `user_id`, `limit`
- 流程：全文/向量检索 messages → 取 top-k 片段 → **二次 LLM 摘要** → 返回短文本
- 不默认注入 prompt；Agent 判断需要回忆时才调用

**二次摘要所用 LLM**

- 宜配置独立 role（如规划中的 `memory_summarizer_llm`），或固定使用 `router_llm`；**避免**使用带 `fallback_chain` 的 `main_llm`，以免章节生成降级链路与会话摘要互相影响。
- 摘要失败可降级为「截断原文片段」返回，不阻塞主对话；详见 `ARCHITECTURE.md` 第 9、11、13 节。

### 4.3 与 LangGraph Checkpointer 的关系

- **Checkpointer**：保存图状态快照（节点、并行 worker 进度），用于恢复与调试
- **Session Archive**：保存完整对话与工具审计，用于跨轮语义搜索
- 二者互补，不互相替代

**并行子任务（见 `ARCHITECTURE.md` 9.7 节）**

- L7 `Send` 分支使用 `thread_id = f"{session_id}:{task_id}"`；Checkpointer 按 thread 隔离，**不**把并行 worker 的全量 messages merge 进父 thread。
- `GraphState` 工作字段（如 `search_batch`、`document_title`）与 Hermes L1 热记忆分离；并行仅提交摘要或 `EvidenceBundle` 引用写入共享 state（Reducer 合并列表字段）。
- `persist_turn` 可在子任务结束时批量/异步刷 archive，避免 N 个 worker 争用同一条热记忆行（实现期用队列或 per-task 缓冲后合并）。

### 4.4 生命周期与合规

- 在线保留：默认 90 天（可配置）
- 冷归档：对象存储 + 索引表
- 删除请求：cascade anonymize messages，保留审计哈希

---

## 5. L3 — Skills（程序性记忆）

### 5.1 字段模型

- `skill_id`, `title`, `version`, `triggers`, `steps`, `tools_used`, `anti_patterns`, `examples`, `success_rate`, `last_used_at`, `tenant_id`

### 5.2 生命周期

```
发现（triggers 匹配意图）
  → 加载摘要（非全文）注入 user message
  → SkillExecutor 按 steps 调用 ToolPort / MCPPort
  → 记录 outcome
  → 成功且复杂度 > 阈值 → SkillExtractor 生成草稿
  → Reviewer 或人工审核 → 发布到 skills/
  → 失败 → 降低 success_rate 或归档
```

### 5.3 与 Cursor Skills 的关系

- 开发态：`.cursor/skills/` 或用户 skills 目录编写
- 运行态：CI 同步到 `runtime/skills/`，`SkillPort` 只读且版本锁定
- 详见 `TOOLS_SKILLS_MCP_DESIGN.md`

### 5.4 工具 `skill_search`

- 输入：`query`, `limit`
- 输出：`SkillSummary` 列表（title + 3–5 行摘要 + skill_id）
- 禁止一次性加载所有 Skill 全文进 prompt

---

## 6. L4 — External Provider（可选）

### 6.1 能力

- 实体解析：同一用户多种称呼、部门别名
- 跨会话结构化事实：职位、部门、项目成员关系
- 对接企业 CRM、工单、LDAP、HR 系统

### 6.2 与 L1 的协作

- L4 拉取的事实**不**整表写入 prompt
- 经摘要流水线更新 USER 的少量行（会话结束后）
- L4 与 L1 冲突时，以 L4 为准并触发 USER 摘要刷新

### 6.3 接口

```python
class ExternalMemoryProvider(Protocol):
    async def resolve_entity(self, mention: str, ctx: RequestContext) -> Entity: ...
    async def fetch_profile_facts(self, user_id: str) -> list[Fact]: ...
```

---

## 7. LangGraph 组件映射

- **L1** → `langgraph.store.BaseStore` + 会话启动加载到 system prompt
- **L2** → 独立 `SessionArchive` 表 + `session_search` 工具
- **L2 辅助** → `Checkpointer` 保存 `GraphState` 快照
- **L3** → Store namespace `("skills", tenant_id)` 或 `skills/` 目录
- **L4** → 可插拔 `ExternalMemoryProvider` 注册到 `MemoryPort`
- **工作记忆** → `GraphState` 字段（`document_title`, `search_batch`, `observation` 等）

### GraphState 建议扩展

- `memory_snapshot_hash: NotRequired[str]`
- `pending_memory_delta: NotRequired[list]`
- `session_id`, `user_id`, `tenant_id`：必填贯穿全图

---

## 8. MemoryPort 接口契约

```python
class MemoryPort(Protocol):
    def compose_prompt_snapshot(self, ctx: RequestContext) -> PromptMemorySnapshot: ...

    async def persist_turn(self, ctx: RequestContext, turn: TurnRecord) -> None: ...

    async def update_prompt_memory(
        self,
        ctx: RequestContext,
        delta: MemoryDelta,
        require_hitl: bool = True,
    ) -> None: ...

    async def session_search(
        self, query: str, ctx: RequestContext, limit: int = 5
    ) -> str: ...

    async def skill_search(
        self, query: str, ctx: RequestContext, limit: int = 3
    ) -> list[SkillSummary]: ...
```

**PromptMemorySnapshot**

- `memory_text: str` — 拼好的 L1 前缀
- `hash: str` — 用于缓存与审计
- `frozen: bool` — 会话内是否为 true

**TurnRecord**

- `role`, `content`, `tool_calls`, `ts`, `trace_id`

---

## 9. 与 Redis / RAG 的协作

**Redis `sess`（见 RAG_DESIGN.md）**

- 缓存 L2 检索的**摘要结果**，加速多轮指代
- 非 L2 真相源；L2 数据库为准

**RAG**

- 企业知识文档走 RAGPort，不写入 L1/L2
- 用户说「记住我喜欢简短回答」→ L1 USER，不走向量库

---

## 10. 与原「四层记忆」文档的映射

项目 `my_test/企业级多轮对话系统记忆管理方案.md` 中的分层可映射为：

- 用户画像层 → Hermes L1 `USER` + L4 External Provider
- 会话记忆层 → Hermes L2 + LangGraph Checkpointer
- 工作记忆层 → `GraphState` 任务槽位（短期）
- 上下文窗口层 → LLM `messages` 最近 N 轮（非持久）

对外叙事统一使用 **Hermes 四层**；实现保留 GraphState 工作记忆字段。

---

## 11. 安全与合规

- L2/L3 写入前 PII 扫描；敏感字段加密存储
- `session_search` 结果受 `acl` 过滤，不可跨租户
- 用户「删除我的数据」：擦除 L1 USER 行、L2 anonymize、失效 Redis sess、联动 RAG 文档删除

---

## 12. Phase 落地顺序

> **一期见 `ARCHITECTURE.md` 1.3.1、15.1 S7**；Skill 目录 **ADR-1**：`skills/published/`。

1. 实现 L1 Store 加载 + 固定 system 前缀 + `memory_snapshot_hash`
2. 实现 L2 `persist_turn`（SQLite）；`session_search` 为二期
3. Checkpointer 与 Session Archive 职责分离；并行 Send 不 merge 全量 messages 进父 thread
4. （二期）L3 `skill_search` + `skills/published/` + ToolPort 联动
5. （三期）L4 External Provider + CRM 适配器
