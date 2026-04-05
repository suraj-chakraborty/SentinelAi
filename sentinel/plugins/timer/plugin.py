"""
Timer Plugin — sentinel/plugins/timer/plugin.py
────────────────────────────────────────────────
Set, list, and cancel countdown timers with spoken and notification alerts.

Trigger examples:
  "set a timer for 5 minutes"
  "timer for 30 seconds"
  "set a 2 hour timer"
  "cancel all timers"
  "list my timers"
"""

from __future__ import annotations

import re
import threading
import time
import logging
from typing import Dict, Optional

from sentinel.core.plugin_system import PluginBase

logger = logging.getLogger("TimerPlugin")

_DURATION_RE = re.compile(
    r"""
    (?:set\s+(?:a\s+)?)?                # optional "set a"
    (?:timer\s+for\s+|timer:\s*)?       # optional "timer for"
    (\d+(?:\.\d+)?)\s*                  # NUMBER
    (second|seconds|sec|s|
     minute|minutes|min|m|
     hour|hours|hr|h)                   # UNIT
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _parse_duration(command: str) -> Optional[float]:
    """Return duration in seconds, or None if not found."""
    match = _DURATION_RE.search(command)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower()
    if unit in ("second", "seconds", "sec", "s"):
        return amount
    if unit in ("minute", "minutes", "min", "m"):
        return amount * 60
    if unit in ("hour", "hours", "hr", "h"):
        return amount * 3600
    return None


class TimerPlugin(PluginBase):
    """Countdown timer plugin with speak-on-completion support."""

    def __init__(self, orchestrator=None):
        super().__init__(orchestrator)
        self._timers: Dict[int, threading.Timer] = {}
        self._counter = 0
        self._lock = threading.Lock()

    # ── Plugin interface ──────────────────────────────────────────────────────

    def can_handle(self, command: str) -> bool:
        cmd = command.lower()
        return any(kw in cmd for kw in ("timer", "countdown"))

    def handle(self, command: str) -> str:
        cmd = command.lower()
        if any(kw in cmd for kw in ("cancel", "stop timer", "clear timer")):
            return self._cancel_all()
        if any(kw in cmd for kw in ("list", "my timers", "show timers")):
            return self._list_timers()
        return self._set_timer(command)

    def on_load(self):
        logger.info("TimerPlugin loaded.")

    def on_unload(self):
        self._cancel_all()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _set_timer(self, command: str) -> str:
        seconds = _parse_duration(command)
        if seconds is None:
            return (
                "I couldn't figure out the duration. "
                "Try 'set a timer for 5 minutes'."
            )

        with self._lock:
            self._counter += 1
            timer_id = self._counter

        label = self._format_duration(seconds)
        t = threading.Timer(seconds, self._fire, args=(timer_id, label))
        t.daemon = True

        with self._lock:
            self._timers[timer_id] = t
        t.start()

        return f"Timer #{timer_id} set for {label}. I'll let you know when it's done."

    def _fire(self, timer_id: int, label: str):
        with self._lock:
            self._timers.pop(timer_id, None)

        msg = f"Timer #{timer_id} is up! {label} have passed."
        logger.info(msg)

        # Speak via orchestrator TTS if available
        if self.orchestrator and hasattr(self.orchestrator, "speak"):
            try:
                self.orchestrator.speak(msg)
            except Exception as exc:
                logger.warning("Could not speak timer alert: %s", exc)

        # Windows notification
        try:
            from win10toast import ToastNotifier
            ToastNotifier().show_toast("SentinelAI Timer", msg, duration=8, threaded=True)
        except Exception:
            pass

    def _cancel_all(self) -> str:
        with self._lock:
            n = len(self._timers)
            for t in self._timers.values():
                t.cancel()
            self._timers.clear()
        return f"Cancelled {n} timer(s)." if n else "No active timers to cancel."

    def _list_timers(self) -> str:
        with self._lock:
            if not self._timers:
                return "No active timers."
            return f"{len(self._timers)} active timer(s): " + ", ".join(
                f"#{tid}" for tid in self._timers
            )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)} second{'s' if seconds != 1 else ''}"
        if seconds < 3600:
            mins = seconds / 60
            return f"{mins:.0f} minute{'s' if mins != 1 else ''}"
        hrs = seconds / 3600
        return f"{hrs:.1f} hour{'s' if hrs != 1 else ''}"
