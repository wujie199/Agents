import asyncio
import logging
import re
import yaml
from pathlib import Path
from typing import Optional, List, Any, Dict

from core.composition.run_context import RunContext
from core.composition.tool_dispatch import invoke_tool, validate_required_tools
from core.ports.skills import (
    SkillPort,
    SkillDefinition,
    SkillStep,
    SkillExecutionResult,
)


class SimpleSkillAdapter:
    def __init__(self, skills_dir: str = "skills/published"):
        self._skills_dir = Path(skills_dir)
        self._skills: Dict[str, SkillDefinition] = {}
        self._raw: Dict[str, dict] = {}
        self._logger = logging.getLogger(__name__)
        self._load_skills()

    def _load_skills(self) -> None:
        self._skills.clear()
        self._raw.clear()
        if not self._skills_dir.exists():
            return

        for skill_file in self._skills_dir.glob("*/skill.yaml"):
            try:
                skill, raw = self._parse_skill_file(skill_file)
                if skill:
                    self._skills[skill.skill_id] = skill
                    self._raw[skill.skill_id] = raw
            except Exception as e:
                self._logger.warning("Error loading skill %s: %s", skill_file, e)

    def _parse_skill_file(self, file_path: Path) -> tuple[Optional[SkillDefinition], dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not data:
            return None, {}

        steps = [
            SkillStep(
                action=s.get("action", ""),
                tool=s.get("tool"),
                args_template=s.get("args_template"),
                on_failure=s.get("on_failure", "abort"),
            )
            for s in data.get("steps", [])
        ]

        skill = SkillDefinition(
            skill_id=data.get("skill_id", file_path.parent.name),
            version=data.get("version", "1.0.0"),
            title=data.get("title", ""),
            triggers=data.get("triggers", []),
            steps=steps,
            required_tools=data.get("required_tools", []),
            max_duration_seconds=data.get("max_duration_seconds", 300),
            acl=data.get("acl", []),
            tenant_id=data.get("tenant_id"),
        )
        return skill, data

    @staticmethod
    def _skill_visible(skill: SkillDefinition, tenant_id: str) -> bool:
        if skill.tenant_id is None or skill.tenant_id == "":
            return True
        return skill.tenant_id == tenant_id

    @staticmethod
    def _check_skill_acl(skill: SkillDefinition, channel: str) -> bool:
        if not skill.acl:
            return True
        return channel in skill.acl

    @staticmethod
    def _as_run_context(context: Any) -> Optional[RunContext]:
        if isinstance(context, RunContext):
            return context
        request = getattr(context, "request", None)
        tools = getattr(context, "tools", None)
        if request is None or tools is None:
            return None
        return RunContext(
            request=request,
            memory=getattr(context, "memory", None),
            tools=tools,
            mcp=getattr(context, "mcp", None),
            skills=getattr(context, "skills", None),
            rag=getattr(context, "rag", None),
        )

    def search(
        self,
        query: str,
        tenant_id: str,
        limit: int = 3,
    ) -> List[dict]:
        results = []
        query_lower = query.lower()

        for skill in self._skills.values():
            if not self._skill_visible(skill, tenant_id):
                continue

            raw = self.get_raw(skill.skill_id) or {}
            if raw.get("status") == "deprecated":
                continue

            score = 0
            for trigger in skill.triggers:
                if query_lower in trigger.lower():
                    score += 1

            for pattern in raw.get("trigger_regex") or []:
                try:
                    if re.search(pattern, query, re.IGNORECASE):
                        score += 2
                except re.error:
                    self._logger.warning(
                        "Invalid trigger_regex for %s: %s",
                        skill.skill_id,
                        pattern,
                    )

            if query_lower in skill.title.lower():
                score += 2

            summary_text = (raw.get("summary") or skill.title or "").lower()
            if query_lower in summary_text:
                score += 1

            if score > 0:
                results.append(
                    {
                        "skill_id": skill.skill_id,
                        "title": skill.title,
                        "score": score,
                        "triggers": skill.triggers,
                        "summary": raw.get("summary", skill.title),
                        "success_rate": raw.get("success_rate", 1.0),
                        "last_used_at": raw.get("last_used_at"),
                        "tenant_id": skill.tenant_id,
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def get(self, skill_id: str) -> Optional[SkillDefinition]:
        return self._skills.get(skill_id)

    async def run(
        self,
        skill_id: str,
        inputs: dict,
        context: Any,
    ) -> SkillExecutionResult:
        skill = self.get(skill_id)
        if not skill:
            return SkillExecutionResult(
                skill_id=skill_id,
                success=False,
                error=f"Skill not found: {skill_id}",
            )

        run_ctx = self._as_run_context(context)
        if run_ctx is None:
            return SkillExecutionResult(
                skill_id=skill_id,
                success=False,
                error="RunContext with request and tools required",
            )

        request = run_ctx.request
        channel = getattr(request, "channel", "user")
        if not self._check_skill_acl(skill, channel):
            return SkillExecutionResult(
                skill_id=skill_id,
                success=False,
                error=f"ACL denied for skill: {skill_id}",
            )

        tenant_id = getattr(request, "tenant_id", "")
        if not self._skill_visible(skill, tenant_id):
            return SkillExecutionResult(
                skill_id=skill_id,
                success=False,
                error=f"Skill not available for tenant: {tenant_id}",
            )

        if skill.required_tools:
            missing = validate_required_tools(skill.required_tools, run_ctx)
            if missing:
                return SkillExecutionResult(
                    skill_id=skill_id,
                    success=False,
                    error=f"Missing required tools: {', '.join(missing)}",
                )

        timeout = max(1, int(skill.max_duration_seconds or 300))
        try:
            return await asyncio.wait_for(
                self._execute_steps(skill, inputs, run_ctx),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return SkillExecutionResult(
                skill_id=skill_id,
                success=False,
                error=f"Skill execution timed out after {timeout}s",
            )

    async def _execute_steps(
        self,
        skill: SkillDefinition,
        inputs: dict,
        run_ctx: RunContext,
    ) -> SkillExecutionResult:
        outputs: dict = {}
        steps_executed = 0

        for step in skill.steps:
            steps_executed += 1
            if not step.tool:
                outputs[step.action] = {"skipped": True, "reason": "no tool"}
                continue

            args = self._prepare_args(step.args_template or {}, inputs, outputs)
            try:
                result = await invoke_tool(run_ctx, step.tool, args)
                outputs[step.action] = result
            except Exception as e:
                if step.on_failure == "skip":
                    outputs[step.action] = {"error": str(e), "skipped": True}
                    continue
                return SkillExecutionResult(
                    skill_id=skill.skill_id,
                    success=False,
                    outputs=outputs,
                    error=str(e),
                    steps_executed=steps_executed,
                )

        return SkillExecutionResult(
            skill_id=skill.skill_id,
            success=True,
            outputs=outputs,
            steps_executed=steps_executed,
        )

    def _prepare_args(
        self,
        template: dict,
        inputs: dict,
        outputs: dict,
    ) -> dict:
        args = {}
        for key, value in template.items():
            if isinstance(value, str):
                if value.startswith("$inputs."):
                    field = value[8:]
                    args[key] = inputs.get(field)
                elif value.startswith("$outputs."):
                    field = value[9:]
                    args[key] = outputs.get(field)
                else:
                    args[key] = value
            else:
                args[key] = value
        return args

    def list_skills(self, tenant_id: str) -> List[str]:
        return [
            s.skill_id
            for s in self._skills.values()
            if self._skill_visible(s, tenant_id)
        ]

    def validate_tools(
        self,
        skill: SkillDefinition,
        available_tools: List[str],
    ) -> List[str]:
        available = set(available_tools)
        missing = []
        for tool in skill.required_tools:
            if tool.startswith("mcp."):
                parts = tool.split(".", 2)
                if len(parts) >= 2 and f"mcp.{parts[1]}.*" in available:
                    continue
            if tool not in available:
                missing.append(tool)
        return missing

    def reload(self) -> None:
        self._load_skills()

    def get_raw(self, skill_id: str) -> Optional[dict]:
        if skill_id in self._raw:
            return dict(self._raw[skill_id])
        skill_file = self._skills_dir / skill_id / "skill.yaml"
        if not skill_file.exists():
            return None
        with open(skill_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
