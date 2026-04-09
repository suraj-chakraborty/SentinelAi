import logging
import subprocess

logger = logging.getLogger("Brightness")


class Brightness:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._current = 50

    def _set(self, value: int) -> bool:
        value = max(0, min(100, int(value)))
        self._current = value
        try:
            cmd = [
                "powershell",
                "-Command",
                f"(Get-WmiObject -Namespace root\\WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{value})",
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=False)
            return True
        except Exception as e:
            logger.error("Brightness set failed: %s", e)
            return False

    def execute(self, command: str, entity: str) -> str:
        cmd = (command or "").lower()
        if not cmd:
            return "What brightness command would you like me to perform?"

        if "set brightness" in cmd:
            m = None
            import re
            m = re.search(r"set\s+brightness\s+to\s+(\d{1,3})", cmd)
            if m:
                val = int(m.group(1))
                if self._set(val):
                    return f"Brightness set to {val}%"
                else:
                    return "Failed to set brightness."
            return "Please specify brightness as a percent, e.g., set brightness to 60%"

        if "brightness up" in cmd or "increase brightness" in cmd:
            self._set(self._current + 10)
            return f"Increased brightness to {self._current}%"

        if "brightness down" in cmd or "decrease brightness" in cmd:
            self._set(self._current - 10)
            return f"Decreased brightness to {self._current}%"

        return f"Brightness command not recognized: {entity or command}"
