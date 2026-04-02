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

# Emotion → voice style mapping
EMOTION_VOICE_MAP = {
    "Stressed/Excited": ("en-US-GuyNeural", "+30%", "+10%"),    # voice, rate, volume
    "Calm/Sad":         ("en-US-JennyNeural", "-20%", "-5%"),
    "Neutral":          ("en-US-GuyNeural",   "+0%",  "+0%"),
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
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._process_queue, daemon=True)
        self._worker.start()
        self._tts_disabled = False
        logger.info(f"TTSEngine started. edge-tts={'✓' if _EDGE_TTS_AVAILABLE else '✗'}, pygame={'✓' if _PYGAME_AVAILABLE else '✗'}, pyttsx3={'✓' if _PYTTSX3_AVAILABLE else '✗'}")

    def speak(self, text: str, emotion: str = "Neutral", block: bool = False):
        """Queue text for speech. Non-blocking by default."""
        if not text or not text.strip():
            return
        if self._tts_disabled:
            return
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
        if self._queue.empty():
            self.speak(text)

    def _process_queue(self):
        while True:
            text, emotion, force_local = self._queue.get()
            try:
                self._do_speak(text, emotion, force_local=force_local)
            except Exception as e:
                logger.error(f"TTS error: {e}")
            finally:
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

    def _speak_edge(self, text: str, emotion: str):
        """Synthesize speech with edge-tts and play with pygame."""
        voice, rate, volume = EMOTION_VOICE_MAP.get(emotion, EMOTION_VOICE_MAP["Neutral"])
        tmp_file = os.path.join(_TTS_CACHE_DIR, f"tts_{threading.get_ident()}.mp3")

        async def _synthesize():
            communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
            await communicate.save(tmp_file)

        # Run async synthesis in a new event loop (thread-safe)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_synthesize())
        finally:
            loop.close()

        # Playback
        if _PYGAME_AVAILABLE and os.path.exists(tmp_file):
            with self._lock:
                try:
                    pygame.mixer.music.load(tmp_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)
                finally:
                    try:
                        pygame.mixer.music.stop()
                        pygame.mixer.music.unload()
                        os.remove(tmp_file)
                    except Exception:
                        pass
        else:
            # Fallback: use os.startfile or another player
            os.startfile(tmp_file) if os.path.exists(tmp_file) else None

    def _speak_pyttsx3(self, text: str, emotion: str):
        """Fallback: blocking pyttsx3 synthesis."""
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
            pythoncom.CoUninitialize()
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")


# Module-level singleton
_tts_engine: TTSEngine = None

def get_tts() -> TTSEngine:
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine()
    return _tts_engine
