"""
sentinel/core/perception_stream.py
───────────────────────────────────
Phase 1: Omnipresent Multimodal Perception.
Continuously runs in the background, capturing a screenshot every N seconds,
compressing it, and extracting a semantic string of what the user is doing.

Maintains a rolling 15-minute window for context injection.
"""

import time
import threading
import base64
import logging
from typing import List, Dict

try:
    from PIL import ImageGrab
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

logger = logging.getLogger("PerceptionStream")

class PerceptionStream:
    def __init__(self, llm_callback, interval=5.0, max_history=180):
        # 180 captures at 5 sec = 15 minutes of context
        self.llm_callback = llm_callback
        self.interval = interval
        self.max_history = max_history
        self._history: List[Dict[str, str]] = []
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        if not _PIL_OK:
            logger.warning("PIL not installed. Perception stream unavailable.")
            return False
            
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Omnipresent Perception Stream started.")
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Perception Stream stopped.")

    def _loop(self):
        while self._running:
            try:
                desc = self._analyze_current_screen()
                if desc and len(desc) > 5:
                    with self._lock:
                        self._history.append({
                            "timestamp": time.time(),
                            "description": desc
                        })
                        if len(self._history) > self.max_history:
                            self._history.pop(0)
            except Exception as e:
                logger.error(f"Perception stream error: {e}")
            
            time.sleep(self.interval)

    def _analyze_current_screen(self) -> str:
        """Captures screen and sends to Vision API to get a 1-sentence descriptor."""
        try:
            img = ImageGrab.grab()
            img = img.resize((1280, 720))  # Downsample for speed
            import io
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            
            # The llm_callback needs to handle image injection.
            # If the callback supports dicts (like our Gemini Vision wrapper), pass it.
            prompt = "What is the user currently looking at or doing? Describe in 5 words or less. Ignore standard desktop backgrounds."
            
            return self.llm_callback(prompt, image_b64=b64)
            
        except Exception:
            return ""

    def get_recent_context(self, minutes=5) -> str:
        """Returns what the user has been doing recently."""
        with self._lock:
            cutoff = time.time() - (minutes * 60)
            recent = [item["description"] for item in self._history if item["timestamp"] >= cutoff]
        
        if not recent:
            return "No recent visual context available."
            
        # Deduplicate sequential identical statuses
        deduped = []
        for r in recent:
            if not deduped or deduped[-1] != r:
                deduped.append(r)
                
        return " -> ".join(deduped[:10])
