"""
sentinel/commands/dream_command.py
───────────────────────────────────
Voice Command Plugin for Dream State / Autonomous Night Mode.

Handles: "enter dream state", "start dream mode", "wake up",
         "dream status", "morning report", "dream briefing".
"""

import logging

from sentinel.daemons.dream_state import DreamStateEngine

logger = logging.getLogger("DreamCommand")


class DreamCommand:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.engine = DreamStateEngine()

    def execute(self, command: str, entity: str | None = None) -> str:
        cmd = command.lower().strip()

        # 1. Enter dream state
        if any(
            k in cmd
            for k in (
                "enter dream",
                "start dream",
                "dream mode on",
                "night mode",
                "go to sleep",
                "autonomous mode",
                "background mode",
            )
        ):
            result = self.engine.enter_dream()
            if result["success"]:
                return "🌙 Entering Dream State... I'll run codebase audits, gather tech intelligence, self-heal errors, and prepare your morning briefing. Sweet dreams!"
            return f"Cannot enter dream state: {result.get('message', 'unknown')}"

        # 2. Wake up
        if any(k in cmd for k in ("wake up", "exit dream", "dream mode off", "stop dream", "I'm back", "good morning")):
            result = self.engine.wake_up()
            if result["success"]:
                recs = result["report"]["recommendations"]
                recs_text = "\n".join(f"  • {r}" for r in recs)
                return (
                    f"☀️ Good morning! Dream state complete.\n"
                    f"Tasks completed: {result['report']['tasks_completed']} | "
                    f"Failed: {result['report']['tasks_failed']}\n"
                    f"Recommendations:\n{recs_text}"
                )
            return "Unable to wake from dream state."

        # 3. Status
        if any(k in cmd for k in ("dream status", "dream phase", "night mode status")):
            status = self.engine.get_status()
            return (
                f"Dream State: {status['phase'].upper()} | "
                f"Running: {status['running']} | "
                f"Tasks: {status['tasks_completed']}/{status['tasks_total']} completed | "
                f"Report available: {status['has_report']}"
            )

        # 4. Morning report / briefing
        if any(
            k in cmd
            for k in ("morning report", "dream report", "dream briefing", "night briefing", "what happened overnight")
        ):
            report = self.engine.get_latest_report()
            if not report:
                return "No dream report available yet. Enter dream state first with 'enter dream mode'."
            recs = "\n".join(f"  • {r}" for r in report["recommendations"])
            return (
                f"📋 Dream Report ({report['timestamp'][:10]})\n"
                f"Tasks: {report['tasks_completed']} completed, {report['tasks_failed']} failed\n"
                f"Intelligence items: {report['intelligence_count']}\n"
                f"Fix proposals: {report['patches_count']}\n"
                f"Recommendations:\n{recs}"
            )

        return "Dream command not recognized. Try: enter dream state, wake up, dream status, morning report."
