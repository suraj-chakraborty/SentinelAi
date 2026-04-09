"""Formal retrieval context structures for prompt construction.

Phase 3/4: Introduce a clean data carrier for memory, retrieved content,
knowledge content and the current task, and a dedicated prompt builder to
produce LLM prompts in a consistent, testable way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalContext:
    memory_context: str = ""
    retrieved_context: str = ""
    knowledge_context: str = ""
    current_task: str = ""


class PromptBuilder:
    """Build a structured prompt for the LLM from a RetrievalContext."""

    def __init__(self, context: RetrievalContext):
        self.context = context

    def build(self) -> str:
        parts = []
        if self.context.memory_context:
            parts.append(self.context.memory_context)
        if self.context.retrieved_context:
            parts.append("[Retrieved]\n" + self.context.retrieved_context)
        if self.context.knowledge_context:
            parts.append("[Knowledge]\n" + self.context.knowledge_context)
        if self.context.current_task:
            parts.append(f"[Current Task]\n{self.context.current_task}")
        return "\n".join(parts)
