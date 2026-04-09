import logging

logger = logging.getLogger("Notes")


class Notes:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    def execute(self, command: str, entity: str) -> str:
        # Take a quick note and store in long-term memory as a fact
        text = (command or "").strip()
        if not text:
            return "What should I note?"
        note = entity or text
        if not note:
            return "Note content is empty."
        if hasattr(self.orchestrator, 'long_term_memory'):
            try:
                self.orchestrator.long_term_memory.store_fact(note, category='notes')
                return "Note saved."
            except Exception as e:
                logger.error("Failed to store note: %s", e)
        return "Note saved (best effort)."
