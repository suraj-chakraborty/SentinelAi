import logging

logger = logging.getLogger("KbSearch")


class KBSearch:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    def execute(self, command: str, entity: str) -> str:
        # Simple wrapper around knowledge base recall
        kb = getattr(self.orchestrator, 'knowledge_base', None)
        if not kb:
            return "Knowledge base not configured."
        query = (entity or command or "").strip()
        if not query:
            return "What should I search the knowledge base for?"
        if hasattr(kb, 'recall_facts'):
            return kb.recall_facts(query)
        return "Knowledge base interface not available."
