# Tools / Skills / MCP 子系统设计

> **文档体系（四份之一）**  
> 主文档：`ARCHITECTURE.md`（**1.3.1 MVP**、**1.5 ADR-2 摄取边界**；协作 §9；风险 §17）  
> 其它：`RAG_DESIGN.md`、`MEMORY_DESIGN.md`  
> 本文档：L5 ToolPort / SkillPort / MCPPort、文档摄取与 OCR/YOLO（一期以 ToolPort + IngestPort 为主）

## 1. 在分层中的位置

- 均属 **L5 领域能力层**（`ToolPort` / `SkillPort` / `MCPPort`），经 `RunContext` 注入 L6。
- Agent 通过 `ctx.tools.invoke`、`ctx.skills.run`、`ctx.mcp.invoke`（或 ReAct 工具名路由）调用；工具名路由规则见本文档第 6 节。
- L6/L7 **禁止**绕过 Port 直接 `import tools.*`（开发调试除外，需 feature flag）。

---

## 2. 三类能力对比

### Native Tools（原生工具）

- Python 函数，`@tool` 或 `StructuredTool`，进程内执行。
- 示例（本项目）：`read_json_all_title`、`save_result_2_json`、`read_word_2_json`、`read_json_context_by_title`。
- 特点：低延迟、强类型 schema、适合确定性 IO。

### Skills（技能）

- **程序性记忆 + 编排说明**；可顺序或条件调用多个 Native / MCP 工具。
- 形态：`SKILL.md`（人类可读）+ `skill.yaml`（机器可读：triggers、steps、required_tools）。
- 执行：`SkillExecutor` 解析 steps → 调用 `ToolPort` / `MCPPort`。
- 存储与 Hermes L3 共用（见 `MEMORY_DESIGN.md`）。

### MCP（Model Context Protocol）

- 进程外工具服务器：stdio / SSE / HTTP。
- 适合：浏览器自动化、数据库控制台、专有 SaaS、高风险隔离环境。
- 通过 `MCPPort.list_tools()` / `call_tool()` 与 Native 工具统一审计。

---

## 3. ToolPort 设计

### 3.1 注册中心

扩展 `config/tools.yml` 概念：

```yaml
tools:
  - name: read_json_all_title
    type: native
    module: tools.read_json_all_title
    acl: [reader]
    timeout_seconds: 30
  - name: pg_query
    type: mcp
    server: postgres-mcp
    acl: [analyst]
    timeout_seconds: 60
```

### 3.2 调用链

```
Agent LLM 产生 tool_call
  → ToolPort.validate_acl(ctx, tool_name)
  → ToolPort.resolve(tool_name) → native | skill_wrapper | mcp
  → 执行（超时、重试、幂等键）
  → 写入 ToolInvocation 审计
  → 结果封装为 ToolMessage 并入 GraphState.messages
```

### 3.3 企业级策略

**Allowlist**

- 生产环境仅允许注册表中声明的工具；未注册名称直接拒绝。

**参数校验**

- 每个工具绑定 JSON Schema；校验失败不执行。
- 禁止工具参数接受原始 SQL 字符串；结构化查询走 `RAGPort` SQL 路径。

**沙箱**

- 代码执行、任意文件写、网络出站需 HITL 或独立容器沙箱。

**幂等**

- 写操作工具必须支持 `idempotency_key`，防止重试双写。

**并发与批处理**（总规范见 `ARCHITECTURE.md` **9.7 节**）

- **图级并行**：多工具/多章节由 L7 `Send` 拆分，不在 ToolPort 内自建无界线程池。
- **ToolPort**：高风险写操作仍单次 `invoke` + `idempotency_key`；只读类工具可选 `invoke_batch(items[])`，由 Adapter 内 Semaphore 限流。
- **MCPPort**：每 server **连接池** + `mcp_max_inflight_per_server`（`config/concurrency.yml`）；超时与重试在 Adapter 统一实现。
- **PolicyPort**：租户 `max_intra_batch_workers`、QPS 与 MCP 并发共用配额；拒绝时标准错误码，可审计 `degraded`。
- **演进**：`Agents/React_Agent.py` 中 `ConcurrencyController` 逻辑迁入 PolicyPort + `concurrency.yml`，Tool 层不再维护全局 batch 单例。

---

## 4. SkillPort 设计

### 4.1 生命周期

```
触发（triggers 匹配用户意图或 Router 输出）
  → skill_search 返回摘要列表
  → Agent 选定 skill_id
  → SkillExecutor 加载 skill.yaml
  → 按 steps 执行（可含条件分支）
  → 记录 outcome 到 L3 统计
  → 可选：提取新 Skill 草稿 → 审核 → 发布
```

### 4.2 Skill 文档结构（SKILL.md 示例）

```markdown
---
skill_id: search_create_section
version: 1.0.0
triggers:
  - 章节撰写
  - 检索生成
required_tools:
  - read_json_context_by_title
  - vector_retrieve
---

## 目标
根据一级章节与子标题，结合 RAG 证据撰写正文并持久化。

## 步骤
1. 读取章节原文与大纲上下文
2. 调用 RAGPort 获取 EvidenceBundle
3. 调用生成链写入结果 JSON
4. 校验输出结构

## 反模式
- 跳过 RAG 直接编造规范条款
- 未带 tenant_id 写入文件
```

### 4.3 skill.yaml 机器字段

- `skill_id`, `version`, `triggers`（字符串或正则列表）
- `steps`：数组，每项 `{ action, tool, args_template, on_failure }`
- `required_tools`：依赖工具名，启动前检查注册表
- `max_duration_seconds`, `acl`

### 4.4 与 Cursor Skills 同步

- 开发：用户在 `.cursor/skills/` 维护
- CI：`sync_skills.sh` 复制到 `runtime/skills/` 并打版本 tag
- 运行时 `SkillPort` 只读 `runtime/skills/`，禁止 Agent 运行时任意写 skill 目录（提取草稿写 `drafts/` 待审核）

---

## 5. MCPPort 设计

### 5.1 Server Registry（config/mcp_servers.yml）

```yaml
mcp_servers:
  - id: filesystem
    transport: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
    acl: [admin]
  - id: postgres-mcp
    transport: sse
    url: https://internal.example/mcp/postgres
    acl: [analyst]
```

### 5.2 连接管理

- **懒启动**：首次 `list_tools` 时建立连接
- **健康检查**：周期性 ping；失败则从注册表临时摘除并告警
- **工具名冲突**：对外暴露名统一为 `mcp.{server_id}.{tool_name}`

### 5.3 安全

- MCP Server 部署在独立网络段；凭据由密钥服务注入环境变量
- 响应大小上限；大二进制走对象存储 URI 引用，不直灌 LLM
- 审计记录 server_id、tool_name、latency、status、args_hash

---

## 6. 调用约定与工具名路由（无 Gateway）

L6 持有 `RunContext` 上的 Port 引用，**不**使用聚合网关类。ReAct 或 LangChain Tool 列表在 **L6 适配层** 将下列名称映射到对应 Port：

**按名称路由（可在 `composition/port_registry.py` 或 `ToolPort` 内实现）**

- 名称以 `skill.` 前缀 → `ctx.skills`
- 名称以 `mcp.` 前缀 → `ctx.mcp`
- 名称在 `tools.yml` → `ctx.tools`（Native）
- 名称 `session_search` / `skill_search` → `ctx.memory`（MemoryPort 工具化接口）

**直接调用（推荐在角色代码内显式分支，类型更清晰）**

```python
# Worker 示例（概念）
evidence = await ctx.rag.route_and_retrieve(query, ctx.request, plan)
result = await ctx.tools.invoke("read_json_all_title", args, ctx.request)
```

横切（ACL、审计、脱敏）用 **装饰 `ToolPort` 的 Adapter** 或 L4 Middleware，与是否使用 Gateway 无关。

---

## 7. 与 RAG / 记忆的边界

**RAGPort**

- 只读知识检索；不写业务表（反馈表除外）。

**ToolPort**

- 副作用：写文件、调 API、改配置、发消息。

**MemoryPort**

- `session_search`、`skill_search`、更新 L1 prompt 记忆。

**禁止**

- Agent 通过 MCP 绕过 `RAGPort` 直接扫向量库（若 MCP 封装向量服务，须在 adapter 内仍实现 `RAGPort` 接口）。

---

## 8. 观测与审计

每条调用记录：

- `trace_id`, `session_id`, `agent_name`, `capability_type`（native/skill/mcp）
- `name`, `latency_ms`, `status`, `args_hash`

对齐 `Agents/callback_log/handlers.py`：

- 扩展 `on_tool_start` / `on_tool_end` 统一字段
- LangGraph 节点名继续写入 `serialized['name']` 便于对照

---

## 9. 错误处理

**可重试**

- 网络超时、MCP 连接重置：指数退避，最多 N 次

**不可重试**

- ACL 拒绝、Schema 校验失败、业务规则违反：立即返回 ToolMessage error

**Skill 步骤失败**

- 按 `on_failure`：`abort` | `skip` | `fallback_tool` | `ask_human`

---

## 10. 文档摄取与视觉模型（L1，非 Tool）

Word/PDF 等摄取若需 OCR、版面检测，由 **RAG 写入管道**（`rag/ingest`，见 `RAG_DESIGN.md` §2）调用 `get_model("ocr")`、`get_model("yolo")`，不注册为 Agent 默认可调 Tool。模型见 `ARCHITECTURE.md` §13。

**流水线顺序（建议）**

1. 可选 `yolo`：检测版面区域（表格、图、正文块）
2. 按区域或整页调用 `ocr` 得到文本
3. 文本分块后由 `get_model("embedding")` 入库（`RAG_DESIGN.md` 第 14 节）

**OCR 降级**

- 顺序：网关 OCR API → 本地 PaddleOCR → Tesseract（英文为主）
- 全部失败：页级标记 `ocr_status=skipped`，若另有文本层则继续，否则跳过该页

**YOLO 降级**

- 顺序：远程推理 → 本地 Ultralytics/ONNX
- 全部失败：整页送 OCR，不做版面切分

**与 ToolPort 的边界**

- 默认摄取不走 ToolPort；若产品需要对话内「对用户上传图 OCR」，可注册独立 Tool，内部仅 `get_model("ocr")`，并限制 ACL、文件大小与 QPS。

---

## 11. 与本项目现有工具的映射

**读取类**

- `read_word_2_json` — 文档摄入，流水线入口
- `read_json_all_title` — 大纲章节列表
- `read_json_context_by_title` — 按标题取正文

**写入类**

- `save_result_2_json`、`save_result_json` — 生成结果持久化

**注册建议 acl**

- `reader`：只读 JSON / Word 转换
- `writer`：save_result 系列
- `admin`：MCP filesystem 等高风险工具

---

## 12. 推荐目录结构

```text
ports/
  tools.py
  skills.py
  mcp.py
composition/
  factory.py
  run_context.py
  port_registry.py      # 可选：工具名 → Port
adapters/
  tools_native/       # 包装 tools/*.py
  mcp_client/
skills/
  published/
  drafts/
config/
  tools.yml
  mcp_servers.yml
  concurrency.yml       # 与主文档 PolicyPort / 9.7 对齐
```

---

## 13. Phase 落地顺序

> **一期见 `ARCHITECTURE.md` 15.1 S4**；摄取走 IngestPort，**ADR-2**。

1. `ToolPort` 包装现有 `tools/*` + ACL + 审计；`RunContext` 注入替换 Agent 内直接 import
2. `config/tools.yml` 与代码注册表对齐
3. `IngestPort` 封装 Word→JSON（非 Agent 默认可调 Tool）
4. （二期）`SkillPort`：`skills/published/` + `skill_search` + 单 Skill 执行
5. （二期）`MCPPort` 单 server 试点；连接池与 `concurrency.yml`
6. （二期）`invoke_batch` 只读工具
