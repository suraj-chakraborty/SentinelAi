"""
sentinel/app/state.py
──────────────────────
Centralised, thread-safe application state.

All shared mutable state that was previously scattered as module-level
globals in sentinel_ai.py is now owned by a single AppState instance.
Every field is accessed via properties that acquire the internal RLock,
so concurrent reads/writes from the GUI thread, audio thread, agent
thread, and brain-monitor thread are all safe.

Usage
─────
    from sentinel.app.state import get_state
    state = get_state()

    state.listening_blocked = True          # sets with lock
    if state.agent_active: …               # reads with lock
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AppState:
    """
    Single source of truth for all runtime application state.

    Thread-safety: every property acquisition is guarded by `_lock`
    (a reentrant lock so the same thread can read multiple fields
    without deadlocking).
    """

    # Internal re-entrant lock — not serialised
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ── Audio / Wake-word ────────────────────────────────────────────────────
    _listening_blocked_until: float = 0.0
    _recording_active: bool = False
    _last_wake_time: float = 0.0

    # ── Agent ────────────────────────────────────────────────────────────────
    _agent_active: bool = False
    _pending_step: Optional[str] = None

    # ── Entertainment / Media ────────────────────────────────────────────────
    _entertainment_active: bool = False

    # ── Brain / LLM health ───────────────────────────────────────────────────
    _brain_status: str = "Checking…"
    _gemini_status: str = "Unknown"
    _ollama_status: str = "Unknown"

    # ── GUI ──────────────────────────────────────────────────────────────────
    _overlay_visible: bool = False
    _app_running: bool = True

    # ── Conversation / Session ───────────────────────────────────────────────
    _current_emotion: str = "Neutral"
    _last_auth_time: float = 0.0

    # ────────────────────────────────────────────────────────────────────────
    # Properties (all acquire the lock)
    # ────────────────────────────────────────────────────────────────────────

    # listening_blocked_until ────────────────────────────────────────────────
    @property
    def listening_blocked(self) -> bool:
        with self._lock:
            return time.time() < self._listening_blocked_until

    def block_listening(self, seconds: float) -> None:
        with self._lock:
            self._listening_blocked_until = time.time() + seconds

    def unblock_listening(self) -> None:
        with self._lock:
            self._listening_blocked_until = 0.0

    # recording_active ───────────────────────────────────────────────────────
    @property
    def recording_active(self) -> bool:
        with self._lock:
            return self._recording_active

    @recording_active.setter
    def recording_active(self, value: bool) -> None:
        with self._lock:
            self._recording_active = value

    # last_wake_time ─────────────────────────────────────────────────────────
    @property
    def last_wake_time(self) -> float:
        with self._lock:
            return self._last_wake_time

    @last_wake_time.setter
    def last_wake_time(self, value: float) -> None:
        with self._lock:
            self._last_wake_time = value

    # agent_active ───────────────────────────────────────────────────────────
    @property
    def agent_active(self) -> bool:
        with self._lock:
            return self._agent_active

    @agent_active.setter
    def agent_active(self, value: bool) -> None:
        with self._lock:
            self._agent_active = value

    # pending_step ───────────────────────────────────────────────────────────
    @property
    def pending_step(self) -> Optional[str]:
        with self._lock:
            return self._pending_step

    @pending_step.setter
    def pending_step(self, value: Optional[str]) -> None:
        with self._lock:
            self._pending_step = value

    # entertainment_active ───────────────────────────────────────────────────
    @property
    def entertainment_active(self) -> bool:
        with self._lock:
            return self._entertainment_active

    @entertainment_active.setter
    def entertainment_active(self, value: bool) -> None:
        with self._lock:
            self._entertainment_active = value

    # brain_status ───────────────────────────────────────────────────────────
    @property
    def brain_status(self) -> str:
        with self._lock:
            return self._brain_status

    @brain_status.setter
    def brain_status(self, value: str) -> None:
        with self._lock:
            self._brain_status = value

    # gemini_status ──────────────────────────────────────────────────────────
    @property
    def gemini_status(self) -> str:
        with self._lock:
            return self._gemini_status

    @gemini_status.setter
    def gemini_status(self, value: str) -> None:
        with self._lock:
            self._gemini_status = value

    # ollama_status ──────────────────────────────────────────────────────────
    @property
    def ollama_status(self) -> str:
        with self._lock:
            return self._ollama_status

    @ollama_status.setter
    def ollama_status(self, value: str) -> None:
        with self._lock:
            self._ollama_status = value

    # overlay_visible ────────────────────────────────────────────────────────
    @property
    def overlay_visible(self) -> bool:
        with self._lock:
            return self._overlay_visible

    @overlay_visible.setter
    def overlay_visible(self, value: bool) -> None:
        with self._lock:
            self._overlay_visible = value

    # app_running ────────────────────────────────────────────────────────────
    @property
    def app_running(self) -> bool:
        with self._lock:
            return self._app_running

    @app_running.setter
    def app_running(self, value: bool) -> None:
        with self._lock:
            self._app_running = value

    # current_emotion ────────────────────────────────────────────────────────
    @property
    def current_emotion(self) -> str:
        with self._lock:
            return self._current_emotion

    @current_emotion.setter
    def current_emotion(self, value: str) -> None:
        with self._lock:
            self._current_emotion = value

    # last_auth_time ──────────────────────────────────────────────────────────
    @property
    def last_auth_time(self) -> float:
        with self._lock:
            return self._last_auth_time

    @last_auth_time.setter
    def last_auth_time(self, value: float) -> None:
        with self._lock:
            self._last_auth_time = value

    # ────────────────────────────────────────────────────────────────────────
    # Legacy Compatibility Bridge (get_status / set_status)
    # ────────────────────────────────────────────────────────────────────────

    def get_status(self, key: str) -> any:
        """Legacy compatibility: Get a status value by its old string key."""
        key = key.upper()
        with self._lock:
            if key == "LISTENING_PAUSED":
                return self.listening_blocked
            if key == "AGENT_ACTIVE":
                return self.agent_active
            if key == "RECORDING":
                return self.recording_active
            if key == "PEND_STEP":
                return self.pending_step
            if key == "EMOTION":
                return self.current_emotion
            if key == "GEMINI_STATUS":
                return self.gemini_status
            if key == "OLLAMA_STATUS":
                return self.ollama_status
            if key == "BRAIN_STATUS":
                return self.brain_status
            return None

    def set_status(self, key: str, value: any) -> None:
        """Legacy compatibility: Set a status value by its old string key."""
        key = key.upper()
        with self._lock:
            if key == "LISTENING_PAUSED":
                if value is True: self.block_listening(300) # 5 min default
                else: self.unblock_listening()
            elif key == "AGENT_ACTIVE":
                self.agent_active = bool(value)
            elif key == "RECORDING":
                self.recording_active = bool(value)
            elif key == "EMOTION":
                self.current_emotion = str(value)
            elif key == "GEMINI_STATUS":
                self.gemini_status = str(value)
            elif key == "OLLAMA_STATUS":
                self.ollama_status = str(value)
            elif key == "BRAIN_STATUS":
                self.brain_status = str(value)
            elif key in ("SYS_CPU", "SYS_MEM", "SENTINEL_MEM_MB", "STABILITY_WARNING") or "LATENCY" in key:
                pass # Accept these keys but don't perform special logic (for GUI display)
            elif key == "LAST_AUTH_TIME":
                self.last_auth_time = float(value)

    def update_status(self, key: str, value: any) -> None:
        """Legacy alias for set_status."""
        self.set_status(key, value)


# ── Singleton accessor ────────────────────────────────────────────────────────

_app_state: Optional[AppState] = None
_state_lock = threading.Lock()


def get_state() -> AppState:
    """Return the process-wide AppState singleton (created on first call)."""
    global _app_state
    if _app_state is None:
        with _state_lock:
            if _app_state is None:
                _app_state = AppState()
    return _app_state


def reset_state() -> None:
    """Reset to a fresh AppState — useful for unit tests."""
    global _app_state
    with _state_lock:
        _app_state = AppState()
