"""
sentinel/commands/mobile_command.py
───────────────────────────────────
Voice Command Plugin for Phone Bridge & Mobile Hub.
Handles "who called me", "check phone notifications", and phone status.
"""

import logging

from sentinel.core.mobile_hub import MobileHubAgent

logger = logging.getLogger("MobileCommand")


class MobileCommand:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.hub = MobileHubAgent(orchestrator=orchestrator)

    def execute(self, command: str, entity: str | None = None) -> str:
        cmd_lower = command.lower()

        # 1. Who called me / recent phone notifications
        if any(k in cmd_lower for k in ("who called", "missed calls", "phone calls", "recent calls")):
            events = [e for e in self.hub.get_recent_phone_events() if e["type"] in ("call", "incoming_call")]
            if events:
                callers = ", ".join([e["contact"] for e in events])
                return f"Recent calls received from: {callers}."
            return "No recent incoming phone calls logged."

        # 2. Check messages from phone
        if any(k in cmd_lower for k in ("phone messages", "text messages", "sms notifications")):
            events = [e for e in self.hub.get_recent_phone_events() if e["type"] != "call"]
            if events:
                lines = [f"{e['contact']}: '{e['message']}'" for e in events[-2:]]
                return f"Recent messages: {'; '.join(lines)}."
            return "No recent phone messages logged."

        # 3. Mobile Bridge Status
        if any(k in cmd_lower for k in ("phone status", "mobile bridge", "phone connected")):
            return "Mobile Phone Bridge is active and standing by for incoming calls and remote commands."

        return "Mobile command not recognized."
