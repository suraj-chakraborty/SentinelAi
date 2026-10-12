import logging
import re

logger = logging.getLogger("VolumeControl")


class VolumeControl:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._current = 50  # naive in-process model, 0-100

    def _clamp(self, v: int) -> int:
        return max(0, min(100, int(v)))

    def execute(self, command: str, entity: str) -> str:
        cmd = (command or "").lower()
        if not cmd:
            return "What volume command would you like me to perform?"

        # set volume to X percent
        m = re.search(r"set\s+volume\s+to\s+(\d{1,3})\s*(percent|%)", cmd)
        if m:
            val = int(m.group(1))
            val = self._clamp(val)
            self._current = val
            return f"Volume set to {val}%."

        # volume up
        if any(p in cmd for p in ["volume up", "increase volume", "volume+", "vol up"]):
            self._current = self._clamp(self._current + 10)
            return f"Increased volume to {self._current}%"

        # volume down
        if any(p in cmd for p in ["volume down", "decrease volume", "volume-", "vol down"]):
            self._current = self._clamp(self._current - 10)
            return f"Decreased volume to {self._current}%"

        # mute / unmute
        if "mute" in cmd:
            self._current = 0
            return "Muted volume."
        if "unmute" in cmd or "sound on" in cmd:
            if self._current == 0:
                self._current = 50
            return f"Unmuted. Volume at {self._current}%"

        return f"Volume command not recognized: {entity or command}"
