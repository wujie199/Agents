import yaml
from pathlib import Path
from typing import Optional, List, Any, Dict
from core.ports.skills import (
    SkillPort,
    SkillDefinition,
    SkillStep,
    SkillExecutionResult
)


class SimpleSkillAdapter:
    def __init__(self, skills_dir: str = "skills/published"):
        self._skills_dir = Path(skills_dir)
        self._skills: Dict[str, SkillDefinition] = {}
        self._load_skills()
    
    def _load_skills(self) -> None:
        if not self._skills_dir.exists():
            return
        
        for skill_file in self._skills_dir.glob("*/skill.yaml"):
            try:
                skill = self._parse_skill_file(skill_file)
                if skill:
                    self._skills[skill.skill_id] = skill
            except Exception as e:
                print(f"Error loading skill {skill_file}: {e}")
    
    def _parse_skill_file(self, file_path: Path) -> Optional[SkillDefinition]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not data:
            return None
        
        steps = [
            SkillStep(
                action=s.get("action", ""),
                tool=s.get("tool"),
                args_template=s.get("args_template"),
                on_failure=s.get("on_failure", "abort")
            )
            for s in data.get("steps", [])
        ]
        
        return SkillDefinition(
            skill_id=data.get("skill_id", file_path.parent.name),
            version=data.get("version", "1.0.0"),
            title=data.get("title", ""),
            triggers=data.get("triggers", []),
            steps=steps,
            required_tools=data.get("required_tools", []),
            max_duration_seconds=data.get("max_duration_seconds", 300),
            acl=data.get("acl", [])
        )
    
    def search(
        self,
        query: str,
        tenant_id: str,
        limit: int = 3
    ) -> List[dict]:
        results = []
        query_lower = query.lower()
        
        for skill in self._skills.values():
            score = 0
            
            for trigger in skill.triggers:
                if query_lower in trigger.lower():
                    score += 1
            
            if query_lower in skill.title.lower():
                score += 2
            
            if score > 0:
                raw = self.get_raw(skill.skill_id) or {}
                results.append({
                    "skill_id": skill.skill_id,
                    "title": skill.title,
                    "score": score,
                    "triggers": skill.triggers,
                    "summary": raw.get("summary", skill.title),
                    "success_rate": raw.get("success_rate", 1.0),
                    "last_used_at": raw.get("last_used_at"),
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def get(self, skill_id: str) -> Optional[SkillDefinition]:
        return self._skills.get(skill_id)
    
    def run(
        self,
        skill_id: str,
        inputs: dict,
        context: Any
    ) -> SkillExecutionResult:
        skill = self.get(skill_id)
        if not skill:
            return SkillExecutionResult(
                skill_id=skill_id,
                success=False,
                error=f"Skill not found: {skill_id}"
            )
        
        if not hasattr(context, 'tools'):
            return SkillExecutionResult(
                skill_id=skill_id,
                success=False,
                error="ToolPort not available in context"
            )
        
        outputs = {}
        steps_executed = 0
        
        try:
            for step in skill.steps:
                steps_executed += 1
                
                if step.tool:
                    args = self._prepare_args(step.args_template or {}, inputs, outputs)
                    result = context.tools.invoke(step.tool, args, context.request)
                    outputs[step.action] = result
            
            return SkillExecutionResult(
                skill_id=skill_id,
                success=True,
                outputs=outputs,
                steps_executed=steps_executed
            )
            
        except Exception as e:
            return SkillExecutionResult(
                skill_id=skill_id,
                success=False,
                outputs=outputs,
                error=str(e),
                steps_executed=steps_executed
            )
    
    def _prepare_args(
        self,
        template: dict,
        inputs: dict,
        outputs: dict
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
        return list(self._skills.keys())
    
    def validate_tools(
        self,
        skill: SkillDefinition,
        available_tools: List[str]
    ) -> List[str]:
        missing = []
        for tool in skill.required_tools:
            if tool not in available_tools:
                missing.append(tool)
        return missing
    
    def reload(self) -> None:
        self._skills.clear()
        self._load_skills()

    def get_raw(self, skill_id: str) -> Optional[dict]:
        skill_file = self._skills_dir / skill_id / "skill.yaml"
        if not skill_file.exists():
            return None
        with open(skill_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
