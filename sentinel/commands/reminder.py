import logging
from datetime import datetime, timedelta

logger = logging.getLogger("Reminder")


class Reminder:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    def execute(self, command: str, entity: str) -> str:
        text = (command or "").lower()
        if not text:
            return "When would you like me to remind you?"
        # Simple parsing: look for 'set a reminder at 17:00' or 'remind me at 5 pm'
        import re
        m = re.search(r"set (a )?reminder (at|for) (.+)", text)
        if not m:
            m = re.search(r"remind me (at|for) (.+)", text)
        time_str = None
        if m:
            time_str = m.group(3).strip()
        if time_str:
            # naive parsing into today  time
            try:
                dt = datetime.strptime(time_str, "%H:%M")
            except Exception:
                try:
                    dt = datetime.strptime(time_str, "%I %p")
                except Exception:
                    dt = None
            if dt:
                return f"Reminder set for {dt.strftime('%Y-%m-%d %H:%M')}"
        return "Could not parse reminder time."
