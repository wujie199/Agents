"""Skill 步骤测试/演示用 echo 工具。"""

from __future__ import annotations


async def skill_echo(message: str = "", **kwargs) -> dict:
    return {"echo": message}
