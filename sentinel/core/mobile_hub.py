"""
sentinel/core/mobile_hub.py
───────────────────────────
Unified Mobile Hub & Phone Bridge Agent.
Provides remote out-of-home execution, call/SMS triage on the desktop,
and bi-directional mobile notification mirroring.
"""

import logging
import os
import time
from typing import Any

logger = logging.getLogger("MobileHub")


class MobileHubAgent:
    """
    Coordinates phone/PC messaging, call triage, and remote execution.
    """

    def __init__(self, orchestrator=None, auth_token: str | None = None):
        self.orchestrator = orchestrator
        self.auth_token = auth_token or os.getenv("SENTINEL_PHONE_TOKEN", "sentinel_secure_token_1234")
        self.notifications_log: list[dict[str, Any]] = []

    def triage_incoming_notification(self, event_type: str, contact: str, message: str = "") -> str:
        """
        Receives notification from mobile device (incoming call, SMS, WhatsApp)
        and alerts the user via desktop voice and logs.
        """
        event = {"type": event_type.lower(), "contact": contact, "message": message, "timestamp": time.time()}
        self.notifications_log.append(event)
        logger.info(f"Mobile Notification: [{event_type.upper()}] from {contact}: {message}")

        if event["type"] in ("call", "incoming_call"):
            announcement = f"Sir, incoming phone call from {contact}."
        else:
            announcement = f"New message from {contact}: {message}"

        # Vocalize through orchestrator if available
        if self.orchestrator and hasattr(self.orchestrator, "speak_fn"):
            try:
                self.orchestrator.speak_fn(announcement)
            except Exception:
                pass

        return announcement

    def execute_remote_command(self, command: str, token: str) -> dict[str, Any]:
        """
        Executes command received remotely from phone (Telegram/Signal/REST).
        Verifies authentication token before running.
        """
        if token != self.auth_token:
            logger.warning("Rejected unauthorized remote mobile command.")
            return {"success": False, "error": "Unauthorized: Invalid authentication token"}

        cmd_clean = command.strip()
        logger.info(f"Executing authorized remote command: '{cmd_clean}'")

        # 1. System actions
        if cmd_clean in ("/lock", "lock", "lock pc"):
            try:
                import ctypes

                ctypes.windll.user32.LockWorkStation()
                return {"success": True, "message": "Workstation locked remotely."}
            except Exception as e:
                return {"success": False, "error": str(e)}

        if cmd_clean in ("/status", "status", "system status"):
            import psutil

            cpu = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory().percent
            return {"success": True, "message": f"Desktop Online. CPU: {int(cpu)}%, RAM: {int(ram)}%."}

        # 2. General command execution via orchestrator
        if self.orchestrator and hasattr(self.orchestrator, "command_router") and self.orchestrator.command_router:
            try:
                res = self.orchestrator.command_router.run_command(cmd_clean)
                return {"success": True, "result": str(res or "Executed successfully")}
            except Exception as e:
                return {"success": False, "error": str(e)}

        return {"success": True, "message": f"Processed command: {cmd_clean}"}

    def get_recent_phone_events(self, limit: int = 5) -> list[dict[str, Any]]:
        """Returns recent calls and messages received from the phone bridge."""
        return self.notifications_log[-limit:]
