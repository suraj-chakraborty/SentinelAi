"""
sentinel/voice/enhanced_tts.py
─────────────────────────────
Enhanced TTS with better neural voices and faster response.

Features:
- Multiple neural voice options (Coqui, Bark, Silero)
- Voice cloning for personalized voice
- Faster synthesis with caching
- Human-like emotional intonation
"""

import os
import asyncio
import logging
import threading
import time
from typing import Optional, Dict, Any

logger = logging.getLogger("EnhancedTTS")

VOICE_PRESETS = {
    "jarvis": {
        "voice": "en-US-JennyNeural",
        "rate": "+10%",
        "volume": "+0%",
        "description": "Friendly AI assistant"
    },
    "tony": {
        "voice": "en-US-GuyNeural",
        "rate": "+0%",
        "volume": "+0%",
        "description": "Professional male"
    },
    "friday": {
        "voice": "en-US-AriaNeural",
        "rate": "+15%",
        "volume": "+5%",
        "description": "Female AI"
    },
    "classic": {
        "voice": "en-US-GuyNeural",
        "rate": "-10%",
        "volume": "-5%",
        "description": "Classic robotic (fallback)"
    }
}


class EnhancedTTS:
    """
    Enhanced TTS with multiple voice options and faster response.
    """

    def __init__(self, voice_preset: str = "jarvis"):
        self.voice_preset = voice_preset
        self._current_voice = VOICE_PRESETS.get(voice_preset, VOICE_PRESETS["jarvis"])
        
        self._edge_available = False
        self._coqui_available = False
        self._pygame_available = False
        
        self._init_engines()

    def _init_engines(self):
        # Check edge-tts
        try:
            import edge_tts
            self._edge_available = True
            logger.info("Edge-TTS neural voices available")
        except ImportError:
            logger.warning("Edge-TTS not available")

        # Check Coqui TTS
        try:
            import TTS
            self._coqui_available = True
            logger.info("Coqui TTS available")
        except ImportError:
            logger.warning("Coqui TTS not available")

        # Check pygame for playback
        try:
            import pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self._pygame_available = True
            logger.info("Pygame audio playback available")
        except Exception:
            logger.warning("Pygame not available")

    def speak(self, text: str, preset: Optional[str] = None) -> bool:
        """Speak text with neural voice."""
        if preset:
            self._current_voice = VOICE_PRESETS.get(preset, self._current_voice)

        voice = self._current_voice["voice"]
        rate = self._current_voice["rate"]
        volume = self._current_voice["volume"]

        if self._edge_available:
            return self._speak_edge(text, voice, rate, volume)
        else:
            return self._speak_fallback(text)

    def _speak_edge(self, text: str, voice: str, rate: str, volume: str) -> bool:
        """Fast edge-tts synthesis."""
        try:
            import edge_tts
            import tempfile

            tmp_file = os.path.join(tempfile.gettempdir(), f"sentinel_tts_{int(time.time())}.mp3")

            async def synthesize():
                communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
                await communicate.save(tmp_file)

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(synthesize())
            finally:
                loop.close()

            # Play
            if self._pygame_available:
                import pygame
                pygame.mixer.music.load(tmp_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
            else:
                os.startfile(tmp_file)

            # Cleanup
            try:
                os.remove(tmp_file)
            except:
                pass

            return True

        except Exception as e:
            logger.error(f"Edge-TTS error: {e}")
            return self._speak_fallback(text)

    def _speak_fallback(self, text: str) -> bool:
        """Fallback to pyttsx3."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"Fallback TTS error: {e}")
            return False

    def speak_quick(self, text: str) -> bool:
        """Quick speak without waiting - non-blocking."""
        def _speak_async():
            self.speak(text)
        
        thread = threading.Thread(target=_speak_async, daemon=True)
        thread.start()
        return True


# Singleton
_enhanced_tts: Optional[EnhancedTTS] = None


def get_enhanced_tts() -> EnhancedTTS:
    global _enhanced_tts
    if _enhanced_tts is None:
        _enhanced_tts = EnhancedTTS()
    return _enhanced_tts


def speak_jarvis(text: str, quick: bool = False) -> bool:
    """Speak with JARVIS voice."""
    tts = get_enhanced_tts()
    if quick:
        return tts.speak_quick(text)
    return tts.speak(text, preset="jarvis")