"""
sentinel/core/perception_stream.py
───────────────────────────────────
Manual Multimodal Perception.
Triggers on-demand screen captures with privacy safeguards.
"""

import time
import base64
import logging
import threading
from typing import List, Dict, Optional

try:
    from PIL import ImageGrab
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    import win32gui
    _WIN32_OK = True
except ImportError:
    _WIN32_OK = False

logger = logging.getLogger("PerceptionStream")

# ── Privacy Settings ────────────────────────────────────────────────────────

BLOCKED_TITLES = ["bitwarden", "bank", "login", "password", "vault", "stripe", "crypto"]

class PerceptionStream:
    def __init__(self, llm_callback, max_history=180):
        self.llm_callback = llm_callback
        self.max_history = max_history
        self._history: List[Dict[str, str]] = []
        self._lock = threading.Lock()
        self._running = False # Legacy support for status checks

    def is_privacy_safe(self) -> bool:
        """Checks if the currently active window title contains blocked keywords."""
        if not _WIN32_OK:
            return True # Fallback if win32 not available
            
        try:
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd).lower()
            for blocked in BLOCKED_TITLES:
                if blocked in title:
                    logger.warning(f"Privacy Block: Active window '{title}' matches '{blocked}' filter.")
                    return False
            return True
        except Exception:
            return True

    def analyze_now(self, custom_prompt: str = None) -> str:
        """One-shot capture and analysis. Replaces the background loop."""
        if not _PIL_OK:
            return "Error: PIL not installed. Perception unavailable."
            
        if not self.is_privacy_safe():
            return "Privacy Shield Active: I cannot see the screen while a sensitive window is open."
            
        desc = self._analyze_current_screen(custom_prompt)
        if desc:
            with self._lock:
                self._history.append({
                    "timestamp": time.time(),
                    "description": desc
                })
                if len(self._history) > self.max_history:
                    self._history.pop(0)
        return desc

    def _analyze_current_screen(self, custom_prompt: str = None) -> str:
        """Captures screen and sends to Vision API."""
        try:
            img = ImageGrab.grab()
            # Downsample for speed and context window efficiency
            img = img.resize((1280, 720))
            import io
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            
            prompt = custom_prompt or "Describe what the user is doing in 10 words or less. Ignore standard desktop backgrounds."
            return self.llm_callback(prompt, image_b64=b64)
            
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return "Failed to capture the screen."

    def get_recent_context(self, minutes=5) -> str:
        """Returns what the user has been doing recently based on manual captures."""
        with self._lock:
            cutoff = time.time() - (minutes * 60)
            recent = [item["description"] for item in self._history if item["timestamp"] >= cutoff]
        
        if not recent:
            return "No recent visual context available."
            
        deduped = []
        for r in recent:
            if not deduped or deduped[-1] != r:
                deduped.append(r)
                
        return " -> ".join(deduped[:10])

    def start(self): 
        logger.info("Perception Stream initialized in manual mode.")
        return True

    def stop(self):
        pass
