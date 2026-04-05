"""
Volume Plugin — sentinel/plugins/volume/plugin.py
──────────────────────────────────────────────────
Control system audio volume and mute state via pycaw (Windows Core Audio API).
Falls back to nircmd and then key simulation if pycaw is unavailable.

Trigger examples:
  "volume up"
  "volume down"
  "set volume to 50 percent"
  "mute"
  "unmute"
  "louder"
  "quieter"
"""

from __future__ import annotations

import logging
import re
import subprocess

from sentinel.core.plugin_system import PluginBase

logger = logging.getLogger("VolumePlugin")

_PERCENT_RE = re.compile(r'(\d{1,3})\s*(?:percent|%)', re.IGNORECASE)
_STEP = 10    # percent per "up/down" command


def _get_pycaw_session():
    """Return (interface, volume_object) or None."""
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        iface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(iface, POINTER(IAudioEndpointVolume))
        return volume
    except Exception as exc:
        logger.debug("pycaw unavailable: %s", exc)
        return None


class VolumePlugin(PluginBase):
    """System volume control through Windows Core Audio API."""

    def can_handle(self, command: str) -> bool:
        cmd = command.lower()
        return any(kw in cmd for kw in (
            "volume", "louder", "quieter", "mute", "unmute",
            "sound up", "sound down", "increase sound", "decrease sound",
        ))

    def handle(self, command: str) -> str:
        cmd = command.lower()

        # Mute / unmute
        if "unmute" in cmd:
            return self._set_mute(False)
        if "mute" in cmd:
            return self._set_mute(True)

        # Explicit percentage
        match = _PERCENT_RE.search(command)
        if match:
            level = int(match.group(1))
            return self._set_volume(level)

        # Up / Down / Louder / Quieter
        if any(kw in cmd for kw in ("up", "louder", "increase", "higher")):
            return self._adjust_volume(+_STEP)
        if any(kw in cmd for kw in ("down", "quieter", "decrease", "lower")):
            return self._adjust_volume(-_STEP)

        # Just "volume" alone → report current level
        return self._get_volume_str()

    def on_load(self):
        logger.info("VolumePlugin loaded.")

    # ── pycaw implementation ──────────────────────────────────────────────────

    def _get_volume(self) -> int:
        vol = _get_pycaw_session()
        if vol:
            try:
                return int(vol.GetMasterVolumeLevelScalar() * 100)
            except Exception:
                pass
        return -1

    def _get_volume_str(self) -> str:
        level = self._get_volume()
        return f"Volume is at {level}%." if level >= 0 else "Couldn't read current volume."

    def _set_volume(self, level: int) -> str:
        level = max(0, min(100, level))
        vol = _get_pycaw_session()
        if vol:
            try:
                vol.SetMasterVolumeLevelScalar(level / 100.0, None)
                return f"Volume set to {level}%."
            except Exception as exc:
                logger.warning("pycaw set volume failed: %s", exc)
        # Fallback: nircmd
        try:
            subprocess.run(["nircmd", "setsysvolume", str(int(level / 100 * 65535))], check=False)
            return f"Volume set to {level}%."
        except Exception as exc:
            logger.error("nircmd setsysvolume failed: %s", exc)
        return "Sorry, couldn't set the volume."

    def _adjust_volume(self, delta: int) -> str:
        current = self._get_volume()
        if current < 0:
            current = 50    # assume mid if unknown
        return self._set_volume(current + delta)

    def _set_mute(self, muted: bool) -> str:
        vol = _get_pycaw_session()
        if vol:
            try:
                vol.SetMute(1 if muted else 0, None)
                return "Muted." if muted else "Unmuted."
            except Exception as exc:
                logger.warning("pycaw mute failed: %s", exc)
        # Fallback: pyautogui key press
        try:
            import pyautogui
            pyautogui.press("volumemute")
            return "Toggled mute."
        except Exception as exc:
            logger.error("Volume mute fallback failed: %s", exc)
        return "Sorry, couldn't toggle mute."
