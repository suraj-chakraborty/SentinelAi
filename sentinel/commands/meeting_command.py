"""
sentinel/commands/meeting_command.py
───────────────────────────────────
Voice Command Plugin for Meeting Co-Pilot.
Handles starting/stopping meeting listening, capturing action items, and summarizing calls.
"""

import logging
from typing import Optional

logger = logging.getLogger("MeetingCommand")


class MeetingCommand:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._surrogate = None

    def _get_surrogate(self):
        if self._surrogate is None:
            try:
                from sentinel.voice.meeting_surrogate import get_meeting_surrogate
                self._surrogate = get_meeting_surrogate()
            except Exception as e:
                logger.error(f"Failed to get meeting surrogate: {e}")
        return self._surrogate

    def execute(self, command: str, entity: Optional[str] = None) -> str:
        cmd_lower = command.lower()
        surrogate = self._get_surrogate()
        if not surrogate:
            return "Meeting co-pilot service is unavailable."

        # 1. Start meeting
        if any(k in cmd_lower for k in ("start meeting", "join meeting", "listen to meeting", "start recording meeting")):
            ok = surrogate.start_meeting()
            if ok:
                return "Meeting co-pilot active. I am listening to system audio and extracting action items."
            return "Failed to start meeting listening session."

        # 2. End meeting
        if any(k in cmd_lower for k in ("end meeting", "stop meeting", "finish meeting", "leave meeting")):
            summary = surrogate.end_meeting()
            if summary:
                actions = summary.action_items
                action_text = f" Captured {len(actions)} action item(s)." if actions else " No action items detected."
                return f"Meeting ended.{action_text}"
            return "No active meeting session was running."

        # 3. Action items / Summary
        if any(k in cmd_lower for k in ("action items", "meeting summary", "what did we discuss")):
            actions = surrogate._action_items if hasattr(surrogate, "_action_items") else []
            if actions:
                return f"Meeting action items: {'; '.join(actions[:4])}."
            return "No pending action items found from recent meetings."

        return "Meeting command not recognized."
