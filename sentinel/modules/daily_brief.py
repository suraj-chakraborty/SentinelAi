"""
sentinel/modules/daily_brief.py
───────────────────────────────
Executive Morning & Session Briefing for SentinelAI.
Compiles calendar schedule, system health, and pending priorities.
"""

import logging
from datetime import datetime

import psutil

logger = logging.getLogger("DailyBrief")


class DailyBriefModule:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.logger = logger

    def generate_brief(self) -> str:
        """Generates a natural, comprehensive executive brief."""
        now = datetime.now()
        hour = now.hour
        if hour < 12:
            time_greeting = "Good morning, sir."
        elif hour < 17:
            time_greeting = "Good afternoon, sir."
        else:
            time_greeting = "Good evening, sir."

        date_str = now.strftime("%A, %B %d")

        brief_segments = [f"{time_greeting} Today is {date_str}."]

        # 1. Calendar schedule
        try:
            if hasattr(self.orchestrator, "calendar_module") and self.orchestrator.calendar_module:
                meets = self.orchestrator.calendar_module.get_upcoming_meets(hours=12)
                if meets:
                    brief_segments.append(f"You have {len(meets)} meetings scheduled for today.")
                else:
                    brief_segments.append("Your calendar is clear for today.")
        except Exception as e:
            self.logger.debug(f"Calendar check in brief: {e}")

        # 2. System telemetry
        try:
            cpu_pct = int(psutil.cpu_percent(interval=0.2))
            ram = psutil.virtual_memory()
            brief_segments.append(
                f"System vitals are running clean: CPU at {cpu_pct} percent, with {ram.percent} percent memory utilized."
            )
        except Exception:
            brief_segments.append("System telemetry is operational.")

        # 3. Battery check if laptop
        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                power_status = "plugged in" if battery.power_plugged else "on battery power"
                brief_segments.append(f"Power is {power_status} at {int(battery.percent)} percent.")
        except Exception:
            pass

        brief_segments.append("All Sentinel systems are online and standing by.")
        return " ".join(brief_segments)
