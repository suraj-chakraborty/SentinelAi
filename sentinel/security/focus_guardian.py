"""
sentinel/security/focus_guardian.py
───────────────────────────────────
Focus Shield & Ergonomic Health Guardian.
Protects deep work sessions, minimizes distractions, and monitors continuous screen fatigue.
"""

import sys
import os
import time
import threading
import logging
from typing import Optional, List, Dict, Any, Callable

logger = logging.getLogger("FocusGuardian")

DISTRACTION_PROCESS_NAMES = [
    "discord.exe", "steam.exe", "epicgameslauncher.exe",
    "telegram.exe", "spotify.exe", "battle.net.exe"
]


class FocusGuardian:
    """
    Manages Focus Mode (Do Not Disturb + Distraction suppression)
    and Ergonomic continuous activity tracking.
    """

    def __init__(
        self,
        orchestrator=None,
        ergonomic_limit_minutes: int = 60,
        on_ergonomic_alert: Optional[Callable[[str], None]] = None
    ):
        self.orchestrator = orchestrator
        self.ergonomic_limit_minutes = ergonomic_limit_minutes
        self.on_ergonomic_alert = on_ergonomic_alert

        self._in_focus_mode = False
        self._focus_end_time = 0.0

        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._active_streak_seconds = 0.0
        self._last_ergo_alert = 0.0

    @property
    def in_focus_mode(self) -> bool:
        if self._in_focus_mode and time.time() > self._focus_end_time:
            self._in_focus_mode = False
        return self._in_focus_mode

    def get_idle_seconds(self) -> float:
        """Returns seconds since last keyboard or mouse input on Windows."""
        if sys.platform != "win32":
            return 0.0
        try:
            import ctypes
            from ctypes import wintypes

            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

            lii = LASTINPUTINFO()
            lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
            if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
                millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
                return max(0.0, float(millis) / 1000.0)
        except Exception:
            pass
        return 0.0

    def start_focus_mode(self, duration_minutes: int = 45) -> Dict[str, Any]:
        """Activates focus shield: closes/minimizes distractions and suppresses non-critical alerts."""
        self._in_focus_mode = True
        self._focus_end_time = time.time() + (duration_minutes * 60)

        # Minimize or terminate distraction processes
        suppressed = []
        try:
            import psutil
            for proc in psutil.process_iter(['name', 'pid']):
                p_name = proc.info['name'].lower() if proc.info['name'] else ''
                if p_name in DISTRACTION_PROCESS_NAMES:
                    suppressed.append(proc.info['name'])
                    # We terminate or suspend to eliminate distraction
                    try: proc.terminate()
                    except Exception: pass
        except Exception as e:
            logger.debug(f"Process inspection during focus mode: {e}")

        logger.info(f"Focus mode activated for {duration_minutes} min. Suppressed: {suppressed}")
        return {
            "success": True,
            "duration_minutes": duration_minutes,
            "suppressed_apps": suppressed
        }

    def end_focus_mode(self) -> bool:
        """Ends focus shield session."""
        self._in_focus_mode = False
        self._focus_end_time = 0.0
        logger.info("Focus mode deactivated.")
        return True

    def start(self):
        """Start background ergonomic tracking thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._health_loop, daemon=True, name="FocusHealthLoop")
        self._thread.start()
        logger.info("FocusGuardian started.")

    def stop(self):
        """Stop background thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("FocusGuardian stopped.")

    def _health_loop(self):
        while self._running:
            time.sleep(30)
            idle = self.get_idle_seconds()
            now = time.time()

            if idle > 180.0:  # User took a >3 minute break
                self._active_streak_seconds = 0.0
            else:
                self._active_streak_seconds += 30.0

            # Check continuous streak limit (e.g. 60 minutes)
            streak_mins = self._active_streak_seconds / 60.0
            if streak_mins >= self.ergonomic_limit_minutes and (now - self._last_ergo_alert) > 1800.0:
                msg = f"Pardon me, sir. You have been working continuously for {int(streak_mins)} minutes. I recommend a brief hydration and posture break."
                logger.info(f"[HEALTH GUARDIAN] {msg}")
                self._last_ergo_alert = now
                if self.on_ergonomic_alert:
                    try: self.on_ergonomic_alert(msg)
                    except Exception: pass
                elif self.orchestrator and hasattr(self.orchestrator, "speak_fn"):
                    try: self.orchestrator.speak_fn(msg)
                    except Exception: pass


# Singleton accessor
_focus_guardian: Optional[FocusGuardian] = None

def get_focus_guardian(orchestrator=None) -> FocusGuardian:
    global _focus_guardian
    if _focus_guardian is None:
        _focus_guardian = FocusGuardian(orchestrator=orchestrator)
    return _focus_guardian
