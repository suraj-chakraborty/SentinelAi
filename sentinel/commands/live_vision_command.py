"""
sentinel/commands/live_vision_command.py
───────────────────────────────────────
Voice Command Plugin for Continuous Video-Rate Screen Agent.
Handles "start screen watcher", "where is the <element> button", and live coding guidance.
"""

import logging

from sentinel.vision.continuous_screen_agent import ContinuousScreenAgent

logger = logging.getLogger("LiveVisionCommand")


class LiveVisionCommand:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.agent = ContinuousScreenAgent()

    def execute(self, command: str, entity: str | None = None) -> str:
        cmd_lower = command.lower()

        # 1. Start continuous screen agent
        if any(
            k in cmd_lower
            for k in ("start screen watcher", "watch my screen", "continuous vision", "monitor my coding")
        ):
            self.agent.start()
            return "Continuous screen observer active. Monitoring live coding context at 2 FPS."

        # 2. Stop screen agent
        if any(k in cmd_lower for k in ("stop screen watcher", "stop watching screen")):
            self.agent.stop()
            return "Continuous screen observation halted."

        # 3. Locate UI element for walkthrough
        if "where is" in cmd_lower or "highlight" in cmd_lower:
            elem = entity or command.replace("where is", "").replace("highlight", "").strip()
            coords = self.agent.find_ui_element(elem)
            if coords:
                return f"Located '{elem}' at screen coordinates: X={coords[0]}, Y={coords[1]} (width: {coords[2]}px, height: {coords[3]}px)."
            return f"Could not find element '{elem}' on screen."

        return "Continuous vision command not recognized."
