from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime


@dataclass
class AgentTask:
    task_id: str
    intent: str
    payload: dict = field(default_factory=dict)
    parent_task_id: Optional[str] = None
    deadline: Optional[datetime] = None
    priority: int = 0

    def __post_init__(self):
        if not self.task_id:
            raise ValueError("task_id is required")
        if not self.intent:
            raise ValueError("intent is required")

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.deadline is None:
            return False
        now = now or datetime.now()
        return now > self.deadline

    def to_thread_id(self, session_id: str) -> str:
        return f"{session_id}:{self.task_id}"
