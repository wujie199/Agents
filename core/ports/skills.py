from typing import Protocol, Optional, List, Any
from dataclasses import dataclass, field


@dataclass
class SkillStep:
    action: str
    tool: Optional[str] = None
    args_template: Optional[dict] = None
    on_failure: str = "abort"


@dataclass
class SkillDefinition:
    skill_id: str
    version: str
    title: str
    triggers: List[str]
    steps: List[SkillStep]
    required_tools: List[str] = field(default_factory=list)
    max_duration_seconds: int = 300
    acl: List[str] = field(default_factory=list)


@dataclass
class SkillExecutionResult:
    skill_id: str
    success: bool
    outputs: dict = field(default_factory=dict)
    error: Optional[str] = None
    steps_executed: int = 0


class SkillPort(Protocol):
    def search(
        self,
        query: str,
        tenant_id: str,
        limit: int = 3
    ) -> List[dict]:
        ...
    
    def get(self, skill_id: str) -> Optional[SkillDefinition]:
        ...
    
    def run(
        self,
        skill_id: str,
        inputs: dict,
        context: Any
    ) -> SkillExecutionResult:
        ...
    
    def list_skills(self, tenant_id: str) -> List[str]:
        ...
    
    def validate_tools(
        self,
        skill: SkillDefinition,
        available_tools: List[str]
    ) -> List[str]:
        ...
