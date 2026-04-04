import logging

logger = logging.getLogger("AiAgent")


def _should_run_autonomous_agent(goal: str) -> bool:
    """
    Prefer a direct LLM reply for short Q&A; use the agent loop for actionable / multi-step goals.
    """
    g = (goal or "").lower().strip()
    if not g:
        return False
    words = g.split()

    action_markers = (
        "automate", "organize", "install", "download", "open ", "click ",
        "on my screen", "my desktop", "this folder", "run the", "execute ",
        "search for", "find all", "create a file", "fill out", "submit the",
    )
    if any(m in g for m in action_markers):
        return True

    chat_prefixes = (
        "what is ", "what are ", "who is ", "who are ", "why ", "when did ",
        "define ", "explain ", "translate ", "hello", "hi ", "thanks", "thank you",
    )
    if any(g.startswith(p) for p in chat_prefixes):
        return False

    if g.endswith("?") and len(words) < 14:
        return False

    return len(words) >= 8


class AiAgent:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def execute(self, command: str, goal: str) -> str:
        """
        Logic for AI agent tasks and general LLM calls.
        """
        engine = None
        if self.orchestrator and hasattr(self.orchestrator, "agentic_engine"):
            engine = self.orchestrator.agentic_engine

        if engine and _should_run_autonomous_agent(goal):
            logger.info("Dispatching to AgenticEngine for goal: %s", goal)
            return engine.execute_autonomous_goal(goal)

        if self.orchestrator and hasattr(self.orchestrator, "_safe_llm_call"):
            return self.orchestrator._safe_llm_call(command)

        return "AI capabilities are currently offline."
