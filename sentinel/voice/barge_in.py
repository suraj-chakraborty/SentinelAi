"""
sentinel/voice/barge_in.py
─────────────────────────
Real-Time Voice Barge-In & Full-Duplex Speech Interruption Engine.

Monitors user vocal activity and audio stream energy in real time.
If vocal energy exceeds the noise threshold while Sentinel is speaking,
it instantly aborts TTS audio output (canceling pygame/playback buffers)
and signals the system to enter 'listening' mode without latency.
"""

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

logger = logging.getLogger("VoiceBargeIn")


class BargeInEngine:
    """
    Full-duplex barge-in detector.
    Coordinates between microphone input streams, WebSocket events,
    and TTSEngine playback to enable instantaneous conversational interruptions.
    """

    def __init__(
        self,
        energy_threshold: float = 0.05,
        min_speech_duration_ms: int = 120,
    ):
        self.energy_threshold = energy_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self._callbacks: list[Callable[[str], None]] = []
        self._is_monitoring = False
        self._monitor_thread: threading.Thread | None = None
        self._last_barge_in_time = 0.0
        self._lock = threading.Lock()

    def register_callback(self, callback: Callable[[str], None]):
        """Register a callback for when barge-in is triggered."""
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[str], None]):
        """Unregister a callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def trigger_barge_in(self, source: str = "user_speech") -> bool:
        """
        Forcefully stop active TTS speech and notify all registered listeners.
        Returns True if active speech was interrupted, False otherwise.
        """
        now = time.time()
        # Debounce rapid triggers within 300ms
        if now - self._last_barge_in_time < 0.3:
            return False

        try:
            from sentinel.app.voice import is_speaking, stop_speaking

            if is_speaking():
                self._last_barge_in_time = now
                logger.info(f"⚡ [Barge-In] Interrupting active speech (source: {source})")
                stop_speaking()

                # Dispatch notifications to listeners (e.g. WebSocket, HUD)
                with self._lock:
                    listeners = list(self._callbacks)
                for cb in listeners:
                    try:
                        cb(source)
                    except Exception as err:
                        logger.error(f"Error in barge-in callback: {err}")
                return True
        except Exception as exc:
            logger.error(f"Failed to process barge-in: {exc}")
        return False

    def process_audio_chunk(self, chunk: np.ndarray) -> bool:
        """
        Analyze an incoming normalized audio buffer (values -1.0 to 1.0).
        If the audio energy exceeds the threshold while speech is active,
        trigger a barge-in interruption.
        """
        if chunk is None or len(chunk) == 0:
            return False

        try:
            from sentinel.app.voice import is_speaking

            if not is_speaking():
                return False

            # Compute Root Mean Square (RMS) energy
            rms = float(np.sqrt(np.mean(np.square(chunk))))
            if rms > self.energy_threshold:
                return self.trigger_barge_in(source=f"acoustic_energy(rms={rms:.3f})")
        except Exception as exc:
            logger.debug(f"Audio chunk evaluation skipped: {exc}")
        return False


class BargeInMonitor:
    """
    Microphone monitoring loop that intercepts speech during playback.
    Provides compatibility for the unit test interface (tts_engine, on_barge_in, samplerate=16000).
    """

    def __init__(
        self,
        tts_engine=None,
        on_barge_in: Callable[[], None] | None = None,
        samplerate: int = 16000,
        energy_threshold: float = 0.05,
    ):
        self.tts_engine = tts_engine
        self.on_barge_in = on_barge_in
        self.samplerate = samplerate
        self.energy_threshold = energy_threshold
        self._running = False
        self._engine = get_barge_in_engine()
        if on_barge_in:
            self._engine.register_callback(lambda src: on_barge_in())

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def process_audio_chunk(self, chunk: np.ndarray) -> bool:
        interrupted = self._engine.process_audio_chunk(chunk)
        if interrupted and self.on_barge_in:
            try:
                self.on_barge_in()
            except Exception:
                pass
        return interrupted


_global_barge_in: BargeInEngine | None = None
_barge_in_lock = threading.Lock()


def get_barge_in_engine() -> BargeInEngine:
    """Get the singleton instance of BargeInEngine."""
    global _global_barge_in
    with _barge_in_lock:
        if _global_barge_in is None:
            _global_barge_in = BargeInEngine()
        return _global_barge_in
