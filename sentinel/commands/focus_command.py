"""
sentinel/commands/focus_command.py
──────────────────────────────────
Voice Command Plugin for Focus Shield and Health Guardian.
Handles "start focus mode", "stop focus mode", and "ergonomic status".
"""

import logging
import re

from sentinel.security.focus_guardian import get_focus_guardian

logger = logging.getLogger("FocusCommand")


class FocusCommand:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    def execute(self, command: str, entity: str | None = None) -> str:
        cmd_lower = command.lower()
        guardian = get_focus_guardian(self.orchestrator)

        # 1. Stop focus mode (check first to avoid 'focus mode' in 'stop focus mode')
        if any(k in cmd_lower for k in ("stop focus", "end focus", "disable focus")):
            guardian.end_focus_mode()
            return "Focus Shield deactivated. Standard notifications restored."

        # 2. Start focus mode
        if any(k in cmd_lower for k in ("start focus", "enable focus", "focus mode", "deep focus")):
            duration = 45
            match = re.search(r"(\d+)\s*(?:minute|min)", cmd_lower)
            if match:
                duration = int(match.group(1))

            res = guardian.start_focus_mode(duration_minutes=duration)
            suppressed = res.get("suppressed_apps", [])
            sup_text = f" Closed {len(suppressed)} distraction apps." if suppressed else ""
            return f"Focus Shield activated for {duration} minutes.{sup_text} Notifications suppressed."

        # 3. Ergonomic / break status
        if any(k in cmd_lower for k in ("posture", "break status", "fatigue", "how long have i worked")):
            idle = int(guardian.get_idle_seconds())
            active_min = int(guardian._active_streak_seconds / 60)
            return f"You have been actively working for {active_min} minutes. Idle duration: {idle} seconds."

        return "Focus command not recognized."
