import logging

logger = logging.getLogger("Calendar")


class Calendar:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    def execute(self, command: str, entity: str) -> str:
        # Lightweight wrapper around existing CalendarModule if available
        if not self.orchestrator or not getattr(self.orchestrator, 'calendar_module', None):
            return "Calendar integration is not configured."
        # Basic stub: attempt to add a calendar item if possible
        if "schedule" in (command or "").lower() or "add" in (command or "").lower():
            return "Calendar integration requires Google Calendar API setup."
        return "Calendar action not recognised."
