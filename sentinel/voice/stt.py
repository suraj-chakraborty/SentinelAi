"""
sentinel/voice/stt.py
─────────────────────
Speech-to-Text using faster-whisper (local, offline, fast).
Falls back to Google Speech API if faster-whisper is not installed.

Usage:
    from sentinel.voice.stt import transcribe
    text = transcribe("command.wav")
"""

import os
import logging
import time

logger = logging.getLogger("SentinelSTT")

APPDATA_DIR = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi")

# ---------- faster-whisper (local, primary) ----------
try:
    from faster_whisper import WhisperModel
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not installed. Run: pip install faster-whisper. Using Google STT fallback.")

# ---------- Google STT (fallback) ----------
try:
    import speech_recognition as sr
    _GOOGLE_STT_AVAILABLE = True
except ImportError:
    _GOOGLE_STT_AVAILABLE = False

# ---------- Vosk (offline fallback) ----------
try:
    import vosk as _vosk
    _VOSK_AVAILABLE = True
except ImportError:
    _VOSK_AVAILABLE = False

_whisper_model: "WhisperModel" = None
_whisper_model_size = "base.en"  # ~150MB, fast on CPU


def _get_whisper_model() -> "WhisperModel":
    """Lazy-load the Whisper model (loads once, reused for all calls)."""
    global _whisper_model
    if _whisper_model is None and _WHISPER_AVAILABLE:
        model_dir = os.path.join(APPDATA_DIR, "whisper_models")
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Loading Whisper model '{_whisper_model_size}'... (first run may download ~150MB)")
        try:
            _whisper_model = WhisperModel(
                _whisper_model_size,
                device="cpu",
                compute_type="int8",           # quantized for fast CPU inference
                download_root=model_dir
            )
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
    return _whisper_model


def transcribe(wav_path: str) -> str:
    """
    Transcribe audio file to text.
    Priority: faster-whisper (local) → Google STT (online) → Vosk (offline)
    """
    if not os.path.exists(wav_path):
        logger.error(f"Audio file not found: {wav_path}")
        return ""

    # 1. Try Google STT first when network is available for lower latency on short commands
    if _GOOGLE_STT_AVAILABLE and _has_network():
        result = _transcribe_google(wav_path)
        if result:
            return result

    # 2. Try faster-whisper (local, offline)
    if _WHISPER_AVAILABLE:
        result = _transcribe_whisper(wav_path)
        if result:
            return result

    # 3. Try Vosk (offline fallback, needs model downloaded)
    if _VOSK_AVAILABLE:
        result = _transcribe_vosk(wav_path)
        if result:
            return result

    logger.warning("All STT engines failed or unavailable.")
    return ""


def _transcribe_whisper(wav_path: str) -> str:
    """Transcribe using faster-whisper (local)."""
    try:
        model = _get_whisper_model()
        if model is None:
            return ""
        t0 = time.time()
        segments, info = model.transcribe(
            wav_path,
            beam_size=3,
            language="en",
            condition_on_previous_text=False,
            vad_filter=True,               # Built-in VAD to skip silence
            vad_parameters={"min_silence_duration_ms": 500}
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed = round(time.time() - t0, 2)
        if text:
            logger.info(f"Whisper transcribed in {elapsed}s: '{text}'")
        return text.lower()
    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        return ""


def _transcribe_google(wav_path: str) -> str:
    """Transcribe using Google Speech Recognition (online fallback)."""
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        logger.info(f"Google STT: '{text}'")
        return text.lower()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        logger.warning(f"Google STT request error: {e}")
        return ""
    except Exception as e:
        logger.error(f"Google STT error: {e}")
        return ""


def _transcribe_vosk(wav_path: str) -> str:
    """Transcribe using Vosk (fully offline fallback)."""
    import wave, json
    try:
        model_dir = os.getenv("VOSK_MODEL") or os.path.join(APPDATA_DIR, "vosk-model")
        if not os.path.isdir(model_dir):
            return ""
        model = _vosk.Model(model_dir)
        rec = _vosk.KaldiRecognizer(model, 16000)
        wf = wave.open(wav_path, "rb")
        try:
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)
        finally:
            wf.close()
        result = json.loads(rec.Result())
        text = result.get("text", "").strip()
        logger.info(f"Vosk STT: '{text}'")
        return text.lower()
    except Exception as e:
        logger.error(f"Vosk STT error: {e}")
        return ""


def _has_network() -> bool:
    """Quick network connectivity check."""
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except Exception:
        return False


def set_whisper_model_size(size: str):
    """Change Whisper model. Options: tiny.en, base.en, small.en, medium.en"""
    global _whisper_model_size, _whisper_model
    _whisper_model_size = size
    _whisper_model = None  # Force reload on next use
    logger.info(f"Whisper model set to: {size}")
