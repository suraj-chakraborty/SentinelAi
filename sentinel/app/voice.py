"""
sentinel/app/voice.py
──────────────────────
High-level Voice (STT) and Speech (TTS) interface.
Delegates to core voice modules with fallback logic.
"""

import logging
import threading
import pythoncom
import pyttsx3
import speech_recognition as sr
from typing import Optional

# Core voice modules
from sentinel.voice.tts import get_tts
from sentinel.voice.stt import transcribe as stt_transcribe

logger = logging.getLogger("SentinelVoice")

# ── Speech Output (TTS) ─────────────────────────────────────────────────────

def speak(text: str, emotion: str = "Neutral", block: bool = False):
    """
    High-level speak interface. 
    Uses neural Edge-TTS by default, falling back to local pyttsx3.
    """
    if not text or not str(text).strip():
        return

    text = str(text).strip()
    logger.info(f"Speaking: {text[:60]}... [{emotion}]")

    try:
        # Use primary neural TTS
        get_tts().speak(text, emotion=emotion, block=block)
    except Exception as e:
        logger.warning(f"Neural TTS failed: {e}. Falling back to pyttsx3.")
        _speak_fallback(text, emotion, block)

def _speak_fallback(text: str, emotion: str, block: bool):
    """Legacy pyttsx3 fallback for offline/error cases."""
    def _run():
        try:
            pythoncom.CoInitialize()
            engine = pyttsx3.init()
            # Basic emotional mapping for pyttsx3
            if "Stressed" in emotion or "Excited" in emotion:
                engine.setProperty('rate', engine.getProperty('rate') + 50)
            elif "Calm" in emotion or "Sad" in emotion:
                engine.setProperty('rate', max(engine.getProperty('rate') - 30, 80))
            
            engine.say(text)
            engine.runAndWait()
            pythoncom.CoUninitialize()
        except Exception as exc:
            logger.error(f"pyttsx3 fallback error: {exc}")

    if block:
        _run()
    else:
        threading.Thread(target=_run, daemon=True).start()

# ── Speech Recognition (STT) ──────────────────────────────────────────────

def transcribe(filename: str) -> str:
    """
    Transcribe a WAV file to text.
    Chain: Faster-Whisper (local) -> Google STT (cloud) -> ""
    """
    if not filename:
        return ""

    # 1. Try local Faster-Whisper
    try:
        result = stt_transcribe(filename)
        if result:
            logger.info(f"STT (Whisper): {result}")
            return result.lower()
    except Exception as e:
        logger.warning(f"Whisper STT failed: {e}. Falling back to Google.")

    # 2. Try Google STT Fallback
    try:
        r = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        logger.info(f"STT (Google): {text}")
        return text.lower()
    except Exception as e:
        logger.error(f"Google STT fallback failed: {e}")
        return ""
