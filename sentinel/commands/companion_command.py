"""
sentinel/commands/companion_command.py
──────────────────────────────────────
Command plugin to launch, toggle, or control the Transparent Floating Desktop Widget.
Triggers via voice or text: "launch companion", "open floating widget", "toggle companion".
"""

import threading
import logging
from typing import Optional

logger = logging.getLogger("CompanionCommand")


class CompanionCommand:
    """Controls the transparent floating desktop companion overlay."""

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    def execute(self, command: str, entity: Optional[str] = None) -> str:
        cmd_lower = command.lower()

        try:
            from sentinel.app.floating_widget import get_floating_widget, start_floating_widget
            import tkinter as tk

            # Check if toggling or launching
            if any(k in cmd_lower for k in ("launch", "open", "start", "show", "enable", "companion", "widget")):
                def _launch():
                    try:
                        start_floating_widget(self.orchestrator)
                    except Exception as e:
                        logger.error(f"Error launching floating companion: {e}")

                thread = threading.Thread(target=_launch, daemon=True, name="FloatingWidgetThread")
                thread.start()
                return "Desktop Floating Companion launched. Always-on-top overlay active. Toggle with Win+Space or Alt+J."

            elif any(k in cmd_lower for k in ("toggle", "hide", "minimize", "close")):
                widget = get_floating_widget(self.orchestrator)
                widget.toggle_visibility()
                return "Toggled desktop floating companion visibility."

        except Exception as exc:
            logger.error(f"Failed to execute companion command: {exc}")
            return f"Unable to control floating companion: {exc}"

        return "Floating companion command not recognized."
