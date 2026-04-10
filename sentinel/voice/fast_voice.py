"""
sentinel/voice/fast_voice.py
───────────────────────────
Fast Interactive Voice Handler.

Features:
- Parallel processing (speak WHILE executing)
- Quick response mode (prioritize speed)
- Natural conversational flow
- Better screen analysis integration
"""

import logging
import threading
import time
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger("FastVoice")


class FastVoiceHandler:
    """
    Fast voice handler that speaks WHILE executing, not after.
    """

    def __init__(self, speak_fn: Optional[Callable] = None):
        self.speak_fn = speak_fn or self._default_speak
        self._response_templates = {
            "starting": ["Sure, ", "Okay, ", "On it, ", "Starting "],
            "done": ["Done.", "Completed.", "Finished.", "All set."],
            "checking": ["Checking... ", "Looking... ", "One moment... "]
        }

    def _default_speak(self, text: str):
        """Default speak using existing TTS."""
        try:
            from sentinel.app.voice import speak
            speak(text, block=False)
        except:
            pass

    def execute_with_announcement(
        self,
        command: str,
        execute_fn: Callable,
        pre_speak: Optional[str] = None,
        post_speak: Optional[str] = None,
        quick_mode: bool = True
    ) -> Any:
        """
        Execute command while speaking - parallel execution.
        
        Args:
            command: The command being executed
            execute_fn: Function to execute
            pre_speak: What to say BEFORE executing (e.g., "Opening Chrome")
            post_speak: What to say AFTER (e.g., "Done, Chrome is open")
            quick_mode: If True, skip pre_speak for faster response
        """
        result = None
        
        # Start speaking in background (non-blocking)
        speak_thread = None
        if pre_speak and not quick_mode:
            speak_thread = threading.Thread(
                target=self.speak_fn,
                args=(pre_speak,),
                daemon=True
            )
            speak_thread.start()
        
        # Execute in background
        exec_thread = threading.Thread(
            target=lambda: setattr(self, '_exec_result', execute_fn()),
            daemon=True
        )
        exec_thread.start()
        
        # Wait for execution with timeout
        exec_thread.join(timeout=30)
        
        if hasattr(self, '_exec_result'):
            result = self._exec_result
        
        # Speak result after execution
        if post_speak:
            # Speak in new thread to not block
            threading.Thread(
                target=self.speak_fn,
                args=(post_speak,),
                daemon=True
            ).start()
        
        return result

    def quick_response(self, response: str) -> str:
        """Send quick response with minimal delay."""
        # Speak immediately in background
        threading.Thread(
            target=self.speak_fn,
            args=(response,),
            daemon=True
        ).start()
        return response

    def analyze_screen_and_speak(self) -> str:
        """Analyze screen and speak the result quickly."""
        try:
            from sentinel.modules.vision import VisionModule
            vision = VisionModule()
            
            # Speak "analyzing" while processing
            threading.Thread(
                target=self.speak_fn,
                args=("Analyzing your screen.",),
                daemon=True
            ).start()
            
            # Get analysis
            result = vision.analyze_screen()
            
            # Speak result
            if result and len(result) > 10:
                threading.Thread(
                    target=self.speak_fn,
                    args=(result,),
                    daemon=True
                ).start()
                return result
            
            return "Couldn't analyze the screen."
        
        except Exception as e:
            logger.error(f"Screen analysis error: {e}")
            return f"Sorry, I couldn't analyze the screen: {e}"


# Singleton
_fast_voice: Optional[FastVoiceHandler] = None


def get_fast_voice() -> FastVoiceHandler:
    global _fast_voice
    if _fast_voice is None:
        _fast_voice = FastVoiceHandler()
    return _fast_voice


def quick_speak(text: str):
    """Quick speak without waiting."""
    handler = get_fast_voice()
    handler.quick_response(text)


def analyze_screen_fast() -> str:
    """Analyze screen with fast response."""
    handler = get_fast_voice()
    return handler.analyze_screen_and_speak()