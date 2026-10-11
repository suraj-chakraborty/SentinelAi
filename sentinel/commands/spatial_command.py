"""
sentinel/commands/spatial_command.py
────────────────────────────────────
Voice Command Plugin for Room-Scale Spatial IoT.
Handles desk lighting sync and multi-room audio hand-off.
"""

import logging
from typing import Optional

from sentinel.modules.spatial_iot import SpatialIoTCoordinator

logger = logging.getLogger("SpatialCommand")


class SpatialCommand:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.coordinator = SpatialIoTCoordinator()

    def execute(self, command: str, entity: Optional[str] = None) -> str:
        cmd_lower = command.lower()

        # 1. Desk Focus Light Sync
        if any(k in cmd_lower for k in ("desk lighting", "focus lights", "sync desk", "desk lights")):
            res = self.coordinator.sync_desk_focus()
            return f"Desk context synchronized. {res['status']}"

        # 2. Cinema / Movie Mode
        if any(k in cmd_lower for k in ("cinema mode", "movie mode", "theater mode", "dim lights")):
            res = self.coordinator.sync_media_cinema()
            return f"Cinema scene activated. {res['status']}"

        # 3. Audio Hand-Off
        if any(k in cmd_lower for k in ("switch audio to", "hand off audio", "audio to kitchen", "audio to bedroom")):
            target_room = "Kitchen"
            for r in ("kitchen", "bedroom", "living room", "office"):
                if r in cmd_lower:
                    target_room = r.title()
                    break

            res = self.coordinator.hand_off_audio_to_room(target_room)
            return f"Handing off audio stream to {target_room} speaker. {res['status']}"

        return "Spatial IoT command not recognized."
