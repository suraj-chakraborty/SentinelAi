"""
sentinel/commands/screen_vision.py
───────────────────────────────────
Screen Vision Command - Fast screen analysis with quick response.

Handles "analyze my screen", "what is on my screen" commands.
"""

import logging
import threading
import time
from typing import Optional, Dict, Any

logger = logging.getLogger("ScreenVision")


class ScreenVision:
    """
    Fast screen analysis with parallel execution.
    """

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._vision = None

    def _get_vision(self):
        """Get vision module."""
        if self._vision is None:
            try:
                from sentinel.modules.vision import VisionModule
                self._vision = VisionModule()
            except Exception as e:
                logger.error(f"Vision module error: {e}")
        return self._vision

    def execute(self, command: str, entity: Optional[str] = None) -> str:
        """
        Execute screen vision command with fast response.
        """
        # Quick acknowledgment first (parallel execution)
        self._speak_early("Analyzing your screen.")
        
        # Then analyze in background
        def analyze_background():
            try:
                vision = self._get_vision()
                if vision:
                    result = vision.analyze_screen("Describe what is on this screen in detail.")
                    
                    # Speak result after analysis
                    if result and len(result) > 10:
                        self._speak(result)
                    else:
                        self._speak("I couldn't analyze the screen properly.")
                else:
                    self._speak("Vision module not available.")
            except Exception as e:
                logger.error(f"Screen analysis error: {e}")
                self._speak(f"Sorry, I had trouble analyzing the screen.")

        # Run analysis in background
        thread = threading.Thread(target=analyze_background, daemon=True)
        thread.start()

        return "Analyzing your screen..."

    def _speak_early(self, text: str):
        """Speak immediately without waiting."""
        try:
            from sentinel.app.voice import speak
            speak(text, block=False)
        except:
            pass

    def _speak(self, text: str):
        """Speak result."""
        try:
            from sentinel.app.voice import speak
            speak(text, block=False)
        except:
            pass


_screen_vision_command: Optional[ScreenVision] = None


def get_screen_vision_command() -> ScreenVision:
    global _screen_vision_command
    if _screen_vision_command is None:
        _screen_vision_command = ScreenVision()
    return _screen_vision_command