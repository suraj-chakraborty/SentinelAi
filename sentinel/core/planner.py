from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PlanStep:
    id: int
    text: str
    done: bool = False


class Planner:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._plan: List[PlanStep] = []

    def plan(self, goal: str) -> List[PlanStep]:
        """Generate a tiny plan for the given goal. This is intentionally simple for Phase 1."""
        self._plan = [PlanStep(id=1, text=f"Execute: {goal}")]
        return self._plan

    def get_next_step(self) -> Optional[PlanStep]:
        if not self._plan:
            return None
        return self._plan[0]

    def mark_done(self, step_id: int) -> bool:
        if not self._plan:
            return False
        if self._plan[0].id == step_id:
            self._plan.pop(0)
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "plan": [s.text for s in self._plan],
            "completed": 0,
        }
