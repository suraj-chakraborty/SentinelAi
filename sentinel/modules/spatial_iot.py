"""
sentinel/modules/spatial_iot.py
───────────────────────────────
Room-Scale Spatial Intelligence & Multi-Room Audio Coordinator.
Connects desk context (focus vs cinema) to Home Assistant smart home lighting
and coordinates multi-room audio hand-off based on room presence.
"""

import logging
from typing import Any

from sentinel.modules.iot_hub import IoTHubModule

logger = logging.getLogger("SpatialIoT")


class SpatialIoTCoordinator:
    """
    Coordinates smart lighting scenes and multi-room audio speaker routing.
    """

    def __init__(self, iot_hub: IoTHubModule | None = None):
        self.iot = iot_hub or IoTHubModule()
        self._current_room = "Office"

    def sync_desk_focus(self) -> dict[str, Any]:
        """Activates focus lighting scene and bias backlight for deep work."""
        ok, msg = self.iot.run_scene("focus_mode")
        if not ok:
            # Fallback direct light toggle
            ok, msg = self.iot.control_home_assistant("light.desk_lamp", service="turn_on", domain="light")
        logger.info(f"Desk Focus Sync: {msg}")
        return {"success": ok, "scene": "focus_mode", "status": msg}

    def sync_media_cinema(self) -> dict[str, Any]:
        """Dims room lights and activates theater ambiance for video playback."""
        ok, msg = self.iot.run_scene("cinema_mode")
        if not ok:
            ok, msg = self.iot.control_home_assistant("light.room_lights", service="turn_off", domain="light")
        logger.info(f"Cinema Mode Sync: {msg}")
        return {"success": ok, "scene": "cinema_mode", "status": msg}

    def hand_off_audio_to_room(self, room_name: str) -> dict[str, Any]:
        """Hands off speech output to designated room speaker via Home Assistant / BLE."""
        normalized_room = room_name.lower().strip()
        speaker_entity = f"media_player.{normalized_room}_speaker"
        self._current_room = room_name.title()

        ok, msg = self.iot.control_home_assistant(speaker_entity, service="turn_on", domain="media_player")
        logger.info(f"Audio Hand-Off to {self._current_room}: {msg}")
        return {"success": ok, "room": self._current_room, "entity": speaker_entity, "status": msg}
