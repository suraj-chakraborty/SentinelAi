"""
sentinel/voice/tts.py
─────────────────────
Neural Text-to-Speech using edge-tts (Microsoft Azure neural voices).
Falls back gracefully to pyttsx3 if edge-tts is unavailable (offline mode).

Usage:
    from sentinel.voice.tts import TTSEngine
    tts = TTSEngine()
    tts.speak("Hello, I am Sentinel.")
"""

import asyncio
import threading
import logging
import os
import time
import queue
from typing import Tuple

logger = logging.getLogger("SentinelTTS")

# ---------- Neural TTS (edge-tts) ----------
try:
    import edge_tts
    _EDGE_TTS_AVAILABLE = True
except ImportError:
    _EDGE_TTS_AVAILABLE = False
    logger.warning("edge-tts not installed. Run: pip install edge-tts. Falling back to pyttsx3.")

# ---------- Audio playback ----------
try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
    _PYGAME_AVAILABLE = True
except Exception:
    _PYGAME_AVAILABLE = False

try:
    import pyttsx3
    import pythoncom
    _PYTTSX3_AVAILABLE = True
except Exception:
    _PYTTSX3_AVAILABLE = False

# Emotion → voice style mapping (improved)
EMOTION_VOICE_MAP = {
    "Stressed/Excited": ("en-US-JennyNeural", "+40%", "+15%"),    # voice, rate, volume
    "Calm/Sad":         ("en-US-JennyNeural", "-10%", "-5%"),
    "Neutral":          ("en-US-JennyNeural", "+15%", "+5%"),       # Better default
    "Happy":            ("en-US-JennyNeural", "+25%", "+10%"),
    "Excited":          ("en-US-AriaNeural", "+30%", "+10%"),
}

APPDATA_DIR = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi")
_TTS_CACHE_DIR = os.path.join(APPDATA_DIR, "tts_cache")
os.makedirs(_TTS_CACHE_DIR, exist_ok=True)


class TTSEngine:
    """
    Thread-safe, non-blocking TTS engine.
    - Primary:  edge-tts  → natural neural voice (online)
    - Fallback: pyttsx3   → robotic but fully offline
    """

    def __init__(self):
        from sentinel.app.config import load_settings
        settings = load_settings()
        self.preferred_voice = settings.get("voice", "en-US-JennyNeural")
        
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._process_queue, daemon=True)
        self._worker.start()
        self._tts_disabled = False
        self._stop_event = threading.Event()  # For interrupting speech
        self._is_active_speaking = False
        logger.info(f"TTSEngine started. edge-tts={'✓' if _EDGE_TTS_AVAILABLE else '✗'}, pygame={'✓' if _PYGAME_AVAILABLE else '✗'}, pyttsx3={'✓' if _PYTTSX3_AVAILABLE else '✗'}")

    @property
    def engine_type(self) -> str:
        """Returns the active TTS engine type."""
        if _EDGE_TTS_AVAILABLE and _PYGAME_AVAILABLE:
            return "edge-tts (Neural)"
        elif _PYTTSX3_AVAILABLE:
            return "pyttsx3 (Offline SAPI5)"
        return "none"

    @property
    def is_speaking(self) -> bool:
        """Returns True if audio is actively playing or speech is queued."""
        if self._is_active_speaking or not self._queue.empty():
            return True
        try:
            if _PYGAME_AVAILABLE and pygame.mixer.get_init():
                return bool(pygame.mixer.music.get_busy())
        except Exception:
            pass
        return False

    def clear_queue(self):
        """Purge all pending utterances from the queue."""
        with self._queue.mutex:
            self._queue.queue.clear()
            self._queue.all_tasks_done.notify_all()
            self._queue.unfinished_tasks = 0

    def stop_speaking(self):
        """Stop current speech immediately and drop queued phrases."""
        self._stop_event.set()
        self.clear_queue()
        try:
            if _PYGAME_AVAILABLE and pygame.mixer.get_init():
                pygame.mixer.music.stop()
                logger.info("Speech playback stopped and queue cleared")
        except Exception as e:
            logger.error(f"Failed to stop speech: {e}")

    def speak(self, text: str, emotion: str = "Neutral", block: bool = False):
        """Queue text for speech. Non-blocking by default."""
        if not text or not text.strip():
            return
        if self._tts_disabled:
            return
        
        # When user initiates new explicit speech, reset stop event if queue was empty
        if self._queue.empty():
            self._stop_event.clear()

        text = str(text)[:500]
        if _PYTTSX3_AVAILABLE and (block or len(text) <= 40):
            if block:
                self._speak_pyttsx3(text, emotion)
            elif self._queue.qsize() <= 1:
                self._queue.put((text, emotion, True))
            return
        self._queue.put((text, emotion, False))
        if block:
            self._queue.join()

    def speak_low_priority(self, text: str):
        """Speaks only if queue is empty (don't interrupt)."""
        if self._queue.empty() and not self.is_speaking:
            self.speak(text)

    def _process_queue(self):
        while True:
            text, emotion, force_local = self._queue.get()
            try:
                if self._stop_event.is_set():
                    # Dropped due to interruption
                    continue
                self._is_active_speaking = True
                self._do_speak(text, emotion, force_local=force_local)
            except Exception as e:
                logger.error(f"TTS error: {e}")
            finally:
                self._is_active_speaking = False
                self._queue.task_done()

    def _do_speak(self, text: str, emotion: str, force_local: bool = False):
        """Attempt edge-tts first, fall back to pyttsx3."""
        if _EDGE_TTS_AVAILABLE and not force_local:
            try:
                self._speak_edge(text, emotion)
                return
            except Exception as e:
                logger.warning(f"edge-tts failed ({e}), falling back to pyttsx3")

        if _PYTTSX3_AVAILABLE:
            self._speak_pyttsx3(text, emotion)
        else:
            logger.error("No TTS engine available.")

    def get_voice_config(self) -> Tuple[str, str, str, str]:
        """Resolves active voice, pitch, rate, and volume, including custom cloned profiles and mood styling."""
        from sentinel.app.config import load_settings
        settings = load_settings()
        voice = settings.get("voice", self.preferred_voice)
        base_pitch = settings.get("voice_pitch", "+0Hz")
        base_rate = settings.get("voice_rate", "+0%")

        mood_pitch = "+0Hz"
        mood_rate = "+0%"
        volume = "+0%"
        try:
            from sentinel.modules.mood_engine import get_mood_engine
            mood_style = get_mood_engine().get_tts_styling()
            mood_pitch = mood_style.get("pitch", "+0Hz")
            mood_rate = mood_style.get("rate", "+0%")
            volume = mood_style.get("volume", "+0%")
        except Exception:
            pass

        effective_pitch = mood_pitch if mood_pitch != "+0Hz" else base_pitch
        effective_rate = mood_rate if mood_rate != "+0%" else base_rate
        return voice, effective_pitch, effective_rate, volume

    def _speak_edge(self, text: str, emotion: str):
        """Synthesize speech with edge-tts and play with pygame."""
        voice, pitch, rate, volume = self.get_voice_config()

        # Inflect on emotion tag if explicitly passed
        if emotion == "Stressed/Excited":
            rate = "+25%"
        elif emotion == "Calm/Sad":
            rate = "-15%"

        if self._stop_event.is_set():
            return
        
        tmp_file = os.path.join(_TTS_CACHE_DIR, f"tts_{threading.get_ident()}.mp3")

        async def _synthesize():
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
            await communicate.save(tmp_file)

        # Run async synthesis in a new event loop (thread-safe)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_synthesize())
        finally:
            loop.close()

        if self._stop_event.is_set():
            if os.path.exists(tmp_file):
                try: os.remove(tmp_file)
                except Exception: pass
            return

        # Playback
        if _PYGAME_AVAILABLE and os.path.exists(tmp_file):
            with self._lock:
                try:
                    pygame.mixer.music.load(tmp_file)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to complete - no timeout (user can interrupt)
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)
                        # Check if user requested interrupt (via event)
                        if self._stop_event and self._stop_event.is_set():
                            pygame.mixer.music.stop()
                            logger.info("TTS interrupted by user")
                            break
                except Exception as e:
                    logger.warning(f"Pygame playback error: {e}")
                    # Final fallback for this file
                    try: os.startfile(tmp_file)
                    except Exception: pass
                finally:
                    try:
                        pygame.mixer.music.stop()
                        pygame.mixer.music.unload()
                        os.remove(tmp_file)
                    except Exception:
                        pass
        elif os.path.exists(tmp_file):
            # No pygame - use system player
            try:
                os.startfile(tmp_file)
                # We can't easily wait for startfile, so we wait a bit before cleanup
                time.sleep(3.0)
                os.remove(tmp_file)
            except Exception as e:
                logger.error(f"System player fallback failed: {e}")

    def _speak_pyttsx3(self, text: str, emotion: str):
        """Fallback: blocking pyttsx3 synthesis."""
        with self._lock:
            try:
                pythoncom.CoInitialize()
                engine = pyttsx3.init()
                rate = engine.getProperty("rate")
                volume = engine.getProperty("volume")
                if emotion == "Stressed/Excited":
                    engine.setProperty("rate", min(rate + 50, 300))
                    engine.setProperty("volume", min(volume + 0.2, 1.0))
                elif emotion == "Calm/Sad":
                    engine.setProperty("rate", max(rate - 30, 80))
                    engine.setProperty("volume", max(volume - 0.1, 0.3))
                engine.say(text)
                engine.runAndWait()
                # Ensure we clean up the engine engine to release resources
                del engine
                pythoncom.CoUninitialize()
            except Exception as e:
                # If we get 'run loop already started', we just wait a bit and hope the next one works
                logger.error(f"pyttsx3 error: {e}")


# Module-level singleton
_tts_engine: TTSEngine = None

def get_tts() -> TTSEngine:
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine()
    return _tts_engine
