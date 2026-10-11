"""
sentinel/commands/gesture_command.py
─────────────────────────────────────
Voice Command Plugin for Air-Gesture Recognition Control.

Handles: "start gesture control", "stop gesture control",
         "gesture status", "recent gestures".
"""

import logging
from typing import Optional

from sentinel.perception.air_gesture import AirGestureController

logger = logging.getLogger("GestureCommand")


class GestureCommand:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.controller = AirGestureController()

    def execute(self, command: str, entity: Optional[str] = None) -> str:
        cmd = command.lower().strip()

        # 1. Start gesture recognition
        if any(k in cmd for k in ("start gesture", "enable gesture", "gesture on",
                                   "air gesture on", "hand tracking on")):
            success = self.controller.start()
            if success:
                return "✋ Air-gesture recognition activated! I'm now tracking your hand movements via webcam. Swipe, pinch, or show a peace sign to control me."
            return "Gesture recognition is already running."

        # 2. Stop gesture recognition
        if any(k in cmd for k in ("stop gesture", "disable gesture", "gesture off",
                                   "hand tracking off", "air gesture off")):
            self.controller.stop()
            return "Air-gesture recognition deactivated. Webcam released."

        # 3. Status
        if any(k in cmd for k in ("gesture status", "gesture info", "hand tracking status")):
            status = self.controller.get_status()
            return (
                f"Gesture Control: {'Active' if status['running'] else 'Inactive'} | "
                f"Total Gestures: {status['total_gestures_detected']} | "
                f"Last: {status['recent_gesture']} | Camera: {status['camera_index']}"
            )

        # 4. Recent gestures
        if any(k in cmd for k in ("recent gesture", "gesture history", "last gesture")):
            gestures = self.controller.get_recent_gestures(5)
            if not gestures:
                return "No gestures detected yet."
            lines = [f"  • {g['gesture']} → {g['action']} ({g['confidence']:.0%})" for g in gestures]
            return "Recent gestures:\n" + "\n".join(lines)

        return "Gesture command not recognized. Try: start gesture control, gesture status, recent gestures."
