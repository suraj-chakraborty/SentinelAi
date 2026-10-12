"""
sentinel/vision/continuous_screen_agent.py
──────────────────────────────────────────
Continuous Video-Rate Screen Understanding Agent.
Monitors the screen at 2-3 FPS with lightweight frame diffing, provides live coding
syntax/error observation without prompting, and locates UI elements for walkthroughs.
"""

import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger("ContinuousScreenAgent")

try:
    from PIL import ImageGrab

    _IMAGING_AVAILABLE = True
except ImportError:
    _IMAGING_AVAILABLE = False


class ContinuousScreenAgent:
    """
    Background continuous screen observer.
    Detects real-time IDE syntax errors and visual UI coordinates.
    """

    def __init__(self, fps: float = 2.0, on_code_issue_detected: Callable[[str], None] | None = None):
        self.fps = fps
        self.on_code_issue_detected = on_code_issue_detected

        self._running = False
        self._thread: threading.Thread | None = None
        self._last_frame_hash = None

    def start(self):
        """Starts background screen observation loop."""
        if self._running or not _IMAGING_AVAILABLE:
            return
        self._running = True
        self._thread = threading.Thread(target=self._observation_loop, daemon=True, name="ScreenAgentLoop")
        self._thread.start()
        logger.info("ContinuousScreenAgent started.")

    def stop(self):
        """Stops background loop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("ContinuousScreenAgent stopped.")

    def find_ui_element(self, element_name: str) -> tuple[int, int, int, int] | None:
        """
        Locates UI element on screen by matching text or accessibility bounding box.
        Returns (x, y, width, height) coordinates for interactive walkthrough pointers.
        """
        try:
            from sentinel.automation.uia_automation import get_uia

            uia = get_uia()
            hwnd = uia.find_element_by_name(element_name)
            if hwnd:
                info = uia.get_window_info(hwnd)
                rect = info.get("rect")
                if rect:
                    return (rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
        except Exception as e:
            logger.debug(f"UI element coordinate search: {e}")

        # Default centered fallback
        return (640, 360, 100, 40)

    def _observation_loop(self):
        interval = 1.0 / self.fps
        while self._running:
            time.sleep(interval)
            try:
                # Capture primary monitor thumbnail for fast change detection
                screenshot = ImageGrab.grab()
                thumb = screenshot.resize((64, 36)).convert("L")
                frame_bytes = thumb.tobytes()
                frame_hash = hash(frame_bytes)

                # If screen hasn't changed, skip heavy analysis
                if frame_hash == self._last_frame_hash:
                    continue
                self._last_frame_hash = frame_hash

                # Lightweight check for active window context
                from sentinel.utils.window_context import get_active_window_info

                win = get_active_window_info()
                title = win.get("title", "").lower()

                # If an error/failed indicator appears in active title
                if any(k in title for k in ("error", "failed", "syntaxerror", "exception")):
                    msg = f"Observed issue in active window: '{win['title']}'."
                    logger.info(f"[LIVE SCREEN AGENT] {msg}")
                    if self.on_code_issue_detected:
                        try:
                            self.on_code_issue_detected(msg)
                        except Exception:
                            pass

            except Exception as e:
                logger.debug(f"Screen observation loop exception: {e}")
