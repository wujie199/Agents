#!/usr/bin/env python3
"""记忆 L1-L4 控制台验证 REPL（不依赖 Agent / LLM）。

用法:
  cd /path/to/Agents
  python document/memory_chat_repl.py --tenant tenant1 --user user1 --session verify1
  python document/memory_chat_repl.py --tenant tenant1 --user user1 --session verify1 --vector
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.domain.context import RequestContext
from core.composition.run_context import RunContext
from core.composition.tool_dispatch import invoke_tool
from core.ports.memory import MemoryDelta, TurnRecord
from agent_platform.tools.adapters.tool_port_adapter import ToolPortAdapter
from agent_platform.memory.memory_tool_registration import register_memory_tools
from document.query_memory import _build_memory, _skill_run_context

HELP = """
命令（L1-L4 验证）:
  /snapshot                 L1 热记忆快照
  /remember key=value       L1 写入 USER（upsert）

  直接输入文字              L2 写入会话 transcript
  /search <词>              L2 会话检索（启动时加 --vector 更佳）
  /turns                    L2 列出本会话消息

  /skills                   L3 列出已发布技能
  /skill <词>               L3 搜索技能
  /run <skill_id> [json]    L3 执行技能（默认 example 用 {"message":"hello"}）

  /entity <称呼>            L4 实体解析
  /facts                    L4 外部 facts 列表
  /profile                  L4 完整 profile YAML

  /help                     显示本帮助
  /quit                     结束会话（L4→L1 finalize 合并）

示例流程:
  我是张三，负责后端 Alpha 项目
  /search Alpha
  /entity 张三
  /facts
  /skill 报告
  /run example
  /snapshot
  /quit
"""


def _default_skill_inputs(skill_id: str) -> dict[str, Any]:
    if skill_id == "example":
        return {"message": "hello from memory repl"}
    if skill_id == "json_section":
        return {"title": "intro", "content": "repl test"}
    return {}


async def _handle_line(
    line: str,
    *,
    memory: Any,
    ctx: RequestContext,
    run_ctx: RunContext,
    tenant: str,
    user: str,
) -> bool:
    """处理一行输入。返回 False 表示退出 REPL。"""
    if line in ("/quit", "/q", "exit"):
        return False

    if line in ("/help", "?"):
        print(HELP)
        return True

    if line == "/snapshot":
        snap = memory.compose_prompt_snapshot(ctx)
        print(f"\n--- L1 hash={snap.hash} ---\n{snap.memory_text}\n")
        return True

    if line.startswith("/remember "):
        body = line[len("/remember ") :].strip()
        if "=" not in body:
            print("用法: /remember key=value")
            return True
        key, value = body.split("=", 1)
        await memory.apply_memory_delta(
            ctx,
            MemoryDelta(key=key.strip(), value=value.strip(), source="user"),
        )
        print("OK (L1 USER upsert)")
        return True

    if line.startswith("/search "):
        query = line[len("/search ") :].strip()
        if not query:
            print("用法: /search <词>")
            return True
        result = await invoke_tool(
            run_ctx, "session_search", {"query": query, "limit": 5}
        )
        print(f"\n[L2 session_search]\n{result}\n")
        return True

    if line == "/turns":
        rows = await memory.list_turns(ctx, limit=50)
        if not rows:
            print("（本会话暂无消息）")
            return True
        for row in rows:
            content = (row.get("content") or "")[:120]
            print(f"  [{row.get('ts', '')}] {row.get('role')}: {content}")
        return True

    if line == "/skills":
        rows = await memory.list_skills(ctx)
        if not rows:
            print("（无已发布技能）")
            return True
        for row in rows:
            print(f"  {row.get('skill_id')}: {row.get('title')}")
        return True

    if line.startswith("/skill "):
        query = line[len("/skill ") :].strip()
        hits = await memory.skill_search(query, ctx, limit=5)
        if not hits:
            print("（无匹配技能）")
            return True
        for hit in hits:
            print(f"  {hit.skill_id}: {hit.title} — {(hit.summary or '')[:80]}")
        return True

    if line.startswith("/run "):
        rest = line[len("/run ") :].strip()
        parts = rest.split(maxsplit=1)
        if not parts:
            print("用法: /run <skill_id> [json_inputs]")
            return True
        skill_id = parts[0]
        inputs: dict[str, Any] = _default_skill_inputs(skill_id)
        if len(parts) > 1:
            try:
                inputs = json.loads(parts[1])
            except json.JSONDecodeError as exc:
                print(f"inputs JSON 无效: {exc}")
                return True
        result = await memory.run_skill(skill_id, inputs, ctx, run_ctx)
        print(
            json.dumps(
                {
                    "skill_id": result.skill_id,
                    "success": result.success,
                    "steps_executed": result.steps_executed,
                    "outputs": result.outputs,
                    "error": result.error,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return True

    if line.startswith("/entity "):
        mention = line[len("/entity ") :].strip()
        entity = await memory.resolve_entity(mention, ctx)
        if entity is None:
            print("（未解析到实体）")
        else:
            print(
                f"mention={entity.mention} "
                f"id={entity.canonical_id} name={entity.display_name}"
            )
        return True

    if line == "/facts":
        facts = await memory.fetch_profile_facts(tenant, user)
        if not facts:
            print("（无外部画像 facts）")
        else:
            print(json.dumps(facts, ensure_ascii=False, indent=2))
        return True

    if line == "/profile":
        profile = await memory.get_profile(tenant, user)
        if not profile:
            print("（无外部画像文件）")
        else:
            print(yaml.dump(profile, allow_unicode=True, default_flow_style=False))
        return True

    await memory.persist_turn(
        ctx,
        TurnRecord(
            role="user",
            content=line,
            ts=datetime.now().isoformat(),
            trace_id=ctx.trace_id,
        ),
    )
    snap = memory.compose_prompt_snapshot(ctx)
    print(
        f"bot> 已写入 L2。"
        f" L1 hash={snap.hash[:12]}…"
        f" 可用 /search /snapshot /entity /facts /skill 继续验证"
    )
    return True


async def memory_chat_repl(
    tenant: str,
    user: str,
    session: str,
    *,
    data_dir: Path,
    config_dir: str,
    use_vector: bool,
    use_llm: bool,
    force_pg: bool,
) -> int:
    memory = _build_memory(
        data_dir,
        config_dir,
        use_llm=use_llm,
        use_vector=use_vector,
        force_pg=force_pg,
    )
    ctx = RequestContext(
        tenant_id=tenant,
        user_id=user,
        session_id=session,
        trace_id="repl",
        channel="cli",
    )
    run_ctx = _skill_run_context(ctx, memory)

    await memory.ensure_session(ctx)
    print("记忆 L1-L4 REPL（空行跳过，/quit 退出并 finalize）")
    print(f"  tenant={tenant}  user={user}  session={session}")
    print(f"  data_dir={data_dir}  vector={use_vector}")
    print(f"  L4 profile: data/external_profiles/{tenant}/{user}.yaml")
    print(HELP)

    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            line = "/quit"
        if not line:
            continue
        cont = await _handle_line(
            line,
            memory=memory,
            ctx=ctx,
            run_ctx=run_ctx,
            tenant=tenant,
            user=user,
        )
        if not cont:
            break

    await memory.end_session(ctx, finalize=True)
    snap = memory.compose_prompt_snapshot(ctx)
    print(f"\n[done] session ended + L4 merged into L1")
    print(f"hash={snap.hash}\n")
    print(snap.memory_text)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="记忆子系统 L1-L4 交互验证 REPL"
    )
    parser.add_argument("--tenant", default="tenant1")
    parser.add_argument("--user", default="user1")
    parser.add_argument("--session", default="verify1")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument(
        "--vector",
        action="store_true",
        help="启用 L2 向量混合检索",
    )
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--pg", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return asyncio.run(
        memory_chat_repl(
            args.tenant,
            args.user,
            args.session,
            data_dir=args.data_dir,
            config_dir=args.config_dir,
            use_vector=args.vector,
            use_llm=args.use_llm,
            force_pg=args.pg,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
