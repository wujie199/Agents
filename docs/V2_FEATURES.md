# v2 新特性文档

## 概述

v2 在 v1 记忆 + RAG 基础上新增 5 个核心特性，形成完整的「检索→融合→推理→记忆→审计」闭环。

---

## 1. 证据融合（Evidence Fusion）

**模块**：`app/agents/roles/evidence_fusion.py`

当检索意图为 `recall_and_knowledge` 时，session_search（回忆）和 RAG（知识库）并行返回结果，由 `fuse_evidence()` 统一融合：

- **指纹去重**：同内容（前 100 字 MD5 hash）只保留高分条目
- **加权排序**：recall 项 × `fusion_recall_weight`（默认 0.6），RAG 项 × `fusion_rag_weight`（默认 0.4）
- **截断保护**：`max_chars` / `max_items` 限制注入 prompt 的证据量

**配置**（`config/chat.yml`）：
```yaml
recall_rag_fusion: true
fusion_recall_weight: 0.6
fusion_rag_weight: 0.4
```

---

## 2. 冲突检测（Conflict Detector）

**模块**：`app/agents/memory/conflict_detector.py`

L1 记忆写入前检测新旧值冲突，支持三种策略：

| 策略 | 行为 |
|------|------|
| `overwrite` | 直接覆盖（无视冲突） |
| `keep_old` | 保留旧值，忽略新值 |
| `ask_user` | 高置信度（≥ `l1_auto_write_confidence_min`）预选新值但仍需 HITL 确认；低置信度保留旧值 |

**接入点**（3 处）：
1. `session_finalize.enrich_l1_before_finalize` — L2→L1 抽取时
2. `name_remember.auto_remember_name_intro` — 姓名自述写入时
3. `enterprise_memory.confirm_pending_l1` — 确认 pending 时

**配置**：
```yaml
l1_conflict_strategy: ask_user
l1_auto_write_confidence_min: 0.9
```

---

## 3. 时间衰减（Time Decay）

**模块**：`agent_platform/memory/adapters/time_decay.py`

公式：`decay_factor = 0.5 ^ (days_since / half_life)`

- 半衰期 90 天时，半年前结果权重约 1/4
- 未来时间戳不衰减（返回 1.0）
- 已接入 `memory_port_adapter` 的 session_search 结果排序

**配置**：
```yaml
time_decay: true
time_decay_half_life_days: 90
```

**REPL 命令**：`/decay-info` 查看配置和衰减预览。

---

## 4. Middleware 洋葱模型

**模块**：`app/agents/middleware/`

洋葱模型包裹图节点（prepare → agent → persist），执行顺序：

```
Tracing.on_enter → Policy.on_enter → Logging.on_enter → Privacy.on_enter
→ [业务节点]
→ Audit.on_exit → Privacy.on_exit → Logging.on_exit → Policy.on_exit → Tracing.on_exit
```

| Middleware | 职责 |
|-----------|------|
| TracingMiddleware | trace_id + span 注入，节点耗时 |
| PolicyMiddleware | ACL / QPS 限流（每租户 100 QPS 默认） |
| LoggingMiddleware | 节点进入/退出/耗时日志 |
| PrivacyMiddleware | PII 检测日志 + `mask_pii()` 脱敏 |
| AuditMiddleware | 关键字段 SHA-256 hash 审计 |

**接入方式**：`app/workflows/chat/graph_def.py` 的 `build_chat_langgraph_workflow` 中自动包装。

---

## 5. DeepAgent 路由门控

**模块**：`app/runtime/adapters/deepagents/`

复杂任务走外层 DeepAgent 规划 harness（任务分解 + 子Agent 调度），简单问题直连内层图。

**判断条件**：
1. 硬规则排除：寒暄/短句（<15字）→ 不走
2. 硬规则命中：多步骤指令正则（`先.*再.*然后`、`帮我写.*报告` 等）→ 走
3. 软规则：LLM 语义判断（可选，`deep_agent_semantic_gate: true`）

**配置**：
```yaml
deep_agent:
  enable: false  # 总开关
  semantic_gate: true
  gate_threshold: 0.7
  planning_model_role: main_llm
  complex_task_patterns:
    - "先.*再.*然后"
    - "帮我写.*报告"
  subagents: []
  interrupt_on:
    delete_file:
      allowed_decisions: [approve, reject]
```

**REPL 命令**：`/todos` 查看 DeepAgent 规划任务列表。

---

## REPL 新增命令

| 命令 | 说明 |
|------|------|
| `/todos` | 查看 DeepAgent 规划任务列表（TodoList） |
| `/conflicts` | 查看 L1 冲突检测结果（pending vs 现有） |
| `/decay-info` | 查看时间衰减配置和效果预览 |
