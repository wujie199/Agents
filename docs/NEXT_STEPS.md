# 后续实施计划

## 当前状态：基础能力层 100% 完成

```
✅ L1 基础设施:  7/7 (100%)
✅ L3 存储:      5/5 (100%)
✅ L5 领域能力:  6/6 (100%)

❌ L6 Agent:     0/3 (0%)
❌ L7 编排:      0/2 (0%)
❌ L8 接入:      0/1 (0%)
```

---

## 需要实现的层次

### 一、L6 Agent 层（角色层）

| 角色 | 职责 | 文件 | 状态 |
|------|------|------|------|
| **Router** | 章节路由、工具选择、retrieval_plan | agents/roles/router.py | ❌ 未实现 |
| **Worker** | RAG 检索、内容生成、结果保存 | agents/roles/worker.py | ❌ 未实现 |
| **Reviewer** | 结果审查、质量控制 | agents/roles/reviewer.py | ❌ 未实现 |

**核心功能**：

```
Router:
├── 分析大纲 → 输出章节列表
├── 选择工具链
├── 生成 retrieval_plan
└── 使用 router_llm（不走 main 的降级链）

Worker:
├── 接收章节任务
├── 调用 RAGPort.route_and_retrieve
├── 调用 main_llm 生成内容
├── 调用 ToolPort 保存结果
└── 返回章节内容

Reviewer:
├── 校验内容格式
├── 检查必需字段
├── 质量评分
└── 决定是否重试
```

---

### 二、L7 应用编排层

| 组件 | 职责 | 文件 | 状态 |
|------|------|------|------|
| **Workflow** | 场景 DAG 定义 | workflows/report_generation/ | ❌ 未实现 |
| **Runtime** | 执行引擎封装 | runtime/adapters/langgraph/ | ❌ 未实现 |

**报告生成场景 DAG**：

```
parse_document      → 解析 Word 文档
    ↓
build_outline       → 构建大纲
    ↓
route_sections      → Router 分析章节
    ↓
dispatch_sections   → 并行 Send 分发
    ↓
[parallel]          → Worker 并行写章节
section_worker × N
    ↓
merge_output        → 合并结果
    ↓
review_output       → Reviewer 审查
    ↓
finalize            → 输出最终报告
```

**关键点**：
- 使用 LangGraph `Send` 实现并行
- 不在节点内使用 `ThreadPoolExecutor`
- `batch_size` 来自 PolicyPort

---

### 三、L8 接入层

| 组件 | 职责 | 文件 | 状态 |
|------|------|------|------|
| **API** | HTTP 入口、鉴权 | api/ | ❌ 未实现 |
| **CLI** | 命令行入口 | main.py | ❌ 未实现 |

**API 功能**：

```
POST /api/v1/report/generate
├── 鉴权 → RequestContext
├── 文件上传 → S3ObjectStore
├── 触发工作流 → WorkflowRuntime
└── 返回 session_id + trace_id

GET /api/v1/report/{session_id}/status
├── 查询工作流状态
└── 返回进度 + 中间结果

GET /api/v1/report/{session_id}/download
├── 下载最终报告
└── 返回文件流
```

---

## 实施优先级

| 阶段 | 层次 | 工作量 | 阻塞关系 |
|------|------|--------|----------|
| **Phase 1** | L6 Agent | 2 天 | 无（基础层已完成） |
| **Phase 2** | L7 编排 | 2 天 | 依赖 L6 |
| **Phase 3** | L8 接入 | 1 天 | 依赖 L7 |

---

## Phase 1: L6 Agent 层（2 天）

### 1.1 Router 角色（0.5 天）

```python
# agents/roles/router.py

async def route_outline(
    outline: dict,
    ctx: RunContext
) -> dict:
    """
    使用 router_llm 分析大纲
    输出：
    - sections: List[Section]
    - retrieval_plan: dict
    - tool_sequence: List[str]
    """
    llm = ctx.get_model("router_llm")
    
    prompt = f"""
    分析以下大纲，输出：
    1. 章节列表
    2. 每个章节的检索策略
    3. 需要调用的工具
    
    大纲：
    {outline}
    """
    
    response = await llm.ainvoke([
        {"role": "system", "content": "你是专业的文档路由分析助手"},
        {"role": "user", "content": prompt}
    ])
    
    return parse_router_output(response)
```

### 1.2 Worker 角色（1 天）

```python
# agents/roles/worker.py

async def section_worker(
    section: Section,
    ctx: RunContext
) -> str:
    """
    单章节生成：
    1. RAG 检索证据
    2. LLM 生成内容
    3. 工具保存结果
    """
    # 1. 检索
    evidence = await ctx.rag.route_and_retrieve(
        query=build_query(section),
        context=ctx.request,
        plan=section.retrieval_plan
    )
    
    # 2. 生成
    llm = ctx.get_model("main_llm")
    
    prompt = f"""
    根据以下证据，撰写章节内容：
    
    章节：{section.title}
    证据：
    {format_evidence(evidence)}
    """
    
    response = await llm.ainvoke([
        {"role": "system", "content": ctx.memory.compose_prompt_snapshot(ctx.request).memory_text},
        {"role": "user", "content": prompt}
    ])
    
    content = response.choices[0].message.content
    
    # 3. 保存
    await ctx.tools.invoke(
        "save_result_2_json",
        {
            "section_id": section.id,
            "content": content
        },
        ctx.request
    )
    
    # 4. 记录记忆
    await ctx.memory.persist_turn(
        ctx.request,
        TurnRecord(
            role="assistant",
            content=content,
            trace_id=ctx.request.trace_id
        )
    )
    
    return content
```

### 1.3 Reviewer 角色（0.5 天）

```python
# agents/roles/reviewer.py

async def review_section(
    content: str,
    section: Section,
    ctx: RunContext
) -> ReviewResult:
    """
    审查章节质量：
    1. 格式校验
    2. 内容完整性
    3. 质量评分
    """
    issues = []
    
    # 格式校验
    if len(content) < 100:
        issues.append("内容过短")
    
    if section.required_keywords:
        missing = [k for k in section.required_keywords if k not in content]
        if missing:
            issues.append(f"缺少关键词: {missing}")
    
    # 质量评分
    llm = ctx.get_model("router_llm")
    
    score_prompt = f"""
    评估以下内容质量（0-10分）：
    
    章节：{section.title}
    内容：{content[:500]}
    """
    
    response = await llm.ainvoke([
        {"role": "user", "content": score_prompt}
    ])
    
    score = parse_score(response)
    
    return ReviewResult(
        passed=len(issues) == 0 and score >= 6,
        score=score,
        issues=issues
    )
```

---

## Phase 2: L7 编排层（2 天）

### 2.1 报告生成工作流（1.5 天）

```python
# workflows/report_generation/graph_def.py

from langgraph import StateGraph, Send

def build_report_workflow():
    graph = StateGraph(ReportState)
    
    # 节点定义
    graph.add_node("parse_document", parse_document_node)
    graph.add_node("build_outline", build_outline_node)
    graph.add_node("route_sections", route_sections_node)
    graph.add_node("section_worker", section_worker_node)
    graph.add_node("merge_output", merge_output_node)
    graph.add_node("finalize", finalize_node)
    
    # 边定义
    graph.add_edge("parse_document", "build_outline")
    graph.add_edge("build_outline", "route_sections")
    
    # 条件边（并行分发）
    graph.add_conditional_edges(
        "route_sections",
        dispatch_sections,  # 返回 List[Send]
        ["section_worker"]
    )
    
    graph.add_edge("section_worker", "merge_output")
    graph.add_edge("merge_output", "finalize")
    
    return graph.compile()

def dispatch_sections(state: ReportState) -> List[Send]:
    """
    并行分发章节到多个 Worker
    """
    batch_size = state.ctx.policy.get_batch_size(
        state.ctx.request.tenant_id
    )
    
    sections = state.sections
    batches = [sections[i:i+batch_size] for i in range(0, len(sections), batch_size)]
    
    sends = []
    for batch in batches:
        for section in batch:
            sends.append(Send(
                "section_worker",
                {"section": section, "batch_id": batch.id}
            ))
    
    return sends
```

### 2.2 Runtime 封装（0.5 天）

```python
# runtime/adapters/langgraph/engine.py

class LangGraphRuntime:
    def __init__(self, checkpointer=None):
        self._checkpointer = checkpointer
    
    def compile(self, graph_def):
        return graph_def.compile(
            checkpointer=self._checkpointer
        )
    
    async def invoke(
        self,
        graph,
        input: dict,
        ctx: RunContext
    ) -> dict:
        config = {
            "configurable": {
                "thread_id": ctx.request.session_id
            }
        }
        
        result = await graph.ainvoke(input, config)
        
        return result
    
    async def stream(
        self,
        graph,
        input: dict,
        ctx: RunContext
    ):
        config = {
            "configurable": {
                "thread_id": ctx.request.session_id
            }
        }
        
        async for event in graph.astream(input, config):
            yield event
```

---

## Phase 3: L8 接入层（1 天）

### 3.1 FastAPI 入口（0.5 天）

```python
# api/app.py

from fastapi import FastAPI, UploadFile, Depends

app = FastAPI()

@app.post("/api/v1/report/generate")
async def generate_report(
    file: UploadFile,
    user_id: str = Depends(get_current_user)
):
    # 1. 构建 RequestContext
    request = RequestContext(
        tenant_id=user_id.tenant_id,
        user_id=user_id.user_id,
        session_id=generate_session_id(),
        trace_id=generate_trace_id(),
        channel="api"
    )
    
    # 2. 注入 Port
    ctx = build_production_context(request)
    
    # 3. 上传文件
    content = await file.read()
    file_key = await ctx.extra["object_store"].upload_async(
        f"upload:{request.session_id}:{file.filename}",
        content
    )
    
    # 4. 触发工作流
    workflow = build_report_workflow()
    runtime = LangGraphRuntime()
    
    result = await runtime.invoke(
        workflow,
        {"file_key": file_key},
        ctx
    )
    
    return {
        "session_id": request.session_id,
        "trace_id": request.trace_id,
        "status": "processing"
    }
```

### 3.2 CLI 入口（0.5 天）

```python
# main.py

import asyncio
import click

@click.command()
@click.option("--input", "-i", required=True, help="输入文档路径")
@click.option("--output", "-o", required=True, help="输出目录")
@click.option("--tenant", "-t", default="default", help="租户ID")
def main(input: str, output: str, tenant: str):
    """报告生成 CLI"""
    
    request = RequestContext(
        tenant_id=tenant,
        user_id="cli_user",
        session_id=generate_session_id(),
        trace_id=generate_trace_id(),
        channel="cli"
    )
    
    ctx = build_development_context(request)
    
    workflow = build_report_workflow()
    runtime = LangGraphRuntime()
    
    result = asyncio.run(
        runtime.invoke(
            workflow,
            {"input_path": input, "output_dir": output},
            ctx
        )
    )
    
    print(f"报告已生成: {result['output_path']}")

if __name__ == "__main__":
    main()
```

---

## 总结

### 后续需要实现

```
Phase 1 (2天): L6 Agent
├── Router 角色 - 章节路由、retrieval_plan
├── Worker 角色 - RAG检索、内容生成
└── Reviewer 角色 - 质量审查

Phase 2 (2天): L7 编排
├── 报告生成工作流 DAG
└── LangGraph Runtime 封装

Phase 3 (1天): L8 接入
├── FastAPI HTTP 入口
└── CLI 命令行入口
```

### 预计工作量

| 阶段 | 工作量 | 交付物 |
|------|--------|--------|
| Phase 1 | 2 天 | L6 Agent 层可运行 |
| Phase 2 | 2 天 | 报告生成场景完整 |
| Phase 3 | 1 天 | API/CLI 可用 |
| **总计** | **5 天** | **完整系统** |

### 建议顺序

```
1. 先实现 Router + Worker（MVP 最小可用）
2. 再实现 Workflow 编排
3. 最后实现 API 接口
```

**是否开始实现 Phase 1 (L6 Agent 层)？**
