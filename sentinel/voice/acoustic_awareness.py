"""
sentinel/voice/acoustic_awareness.py
────────────────────────────────────
Acoustic Awareness Module - Continuous VAD, interrupt detection,
and fluid turn-taking for JARVIS-style interactions.

Replaces strict wake-word pipeline with continuous streaming duplex engine.
"""

import os
import time
import logging
import threading
import queue
import numpy as np
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from sentinel.app.config import APPDATA_DIR

logger = logging.getLogger("AcousticAwareness")

AUDIO_BUFFER_DIR = os.path.join(APPDATA_DIR, "audio_buffers")
os.makedirs(AUDIO_BUFFER_DIR, exist_ok=True)


class VoiceActivityDetector:
    """Silero VAD wrapper for continuous voice activity detection."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._model = None
        self._sample_rate = 16000
        self._init_model()

    def _init_model(self):
        """Load Silero VAD model."""
        try:
            import torch
            self._model = torch.load(os.path.join(os.path.dirname(__file__), "..", "models", "silero_vad.jit"))
            logger.info("Silero VAD loaded")
        except Exception as e:
            logger.warning(f"Silero VAD not available: {e}, using fallback")
            self._model = None

    def is_speaking(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains speech."""
        if self._model is None:
            return self._fallback_vad(audio_chunk)
        
        try:
            import torch
            input_tensor = torch.from_numpy(audio_chunk).float()
            if len(input_tensor) < 512:
                return False
            result = self._model(input_tensor.unsqueeze(0))
            return result.item() > self.threshold
        except Exception as e:
            logger.debug(f"VAD error: {e}")
            return self._fallback_vad(audio_chunk)

    def _fallback_vad(self, audio_chunk: np.ndarray) -> bool:
        """Simple energy-based fallback VAD."""
        if len(audio_chunk) == 0:
            return False
        energy = np.abs(audio_chunk).mean()
        return energy > 0.02


class AcousticAwareness:
    """
    Continuous acoustic awareness with interrupt detection.
    
    Features:
    - Rolling buffer analysis with Silero VAD
    - Interrupt detection during TTS playback
    - Context window management for interruptions
    - Optional stereo direction detection
    """

    def __init__(
        self,
        buffer_size_ms: int = 500,
        vad_threshold: float = 0.5,
        on_interrupt: Optional[Callable] = None,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None
    ):
        self.buffer_size_ms = buffer_size_ms
        self.vad_threshold = vad_threshold
        
        self.on_interrupt = on_interrupt
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        
        self._vad = VoiceActivityDetector(threshold=vad_threshold)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._audio_queue: queue.Queue = queue.Queue(maxsize=20)
        
        self._is_speaking = False
        self._was_speaking = False
        self._speech_start_time: Optional[datetime] = None
        self._interrupt_buffer: list = []
        
        self._context_window: list = []
        self._max_context_items = 10

    def start(self):
        """Start continuous VAD monitoring."""
        if self._running:
            logger.warning("Acoustic awareness already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="AcousticAwareness")
        self._thread.start()
        logger.info("Acoustic awareness started (VAD threshold: {})".format(self.vad_threshold))

    def stop(self):
        """Stop VAD monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Acoustic awareness stopped")

    def feed_audio(self, audio_data: np.ndarray):
        """Feed audio data to the VAD pipeline."""
        try:
            self._audio_queue.put_nowait(audio_data)
        except queue.Full:
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(audio_data)
            except queue.Empty:
                pass

    def _monitor_loop(self):
        """Main VAD monitoring loop."""
        buffer_size = int(16000 * self.buffer_size_ms / 1000)
        
        while self._running:
            try:
                audio_data = self._audio_queue.get(timeout=1)
                
                if len(audio_data) >= buffer_size:
                    chunk = audio_data[-buffer_size:]
                else:
                    chunk = audio_data
                
                is_speaking = self._vad.is_speaking(chunk)
                
                self._process_speech_state(is_speaking, chunk)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"VAD monitoring error: {e}")
                time.sleep(0.1)

    def _process_speech_state(self, is_speaking: bool, audio_chunk: np.ndarray):
        """Process speech state changes and trigger callbacks."""
        self._was_speaking = self._is_speaking
        self._is_speaking = is_speaking
        
        if is_speaking and not self._was_speaking:
            self._speech_start_time = datetime.now()
            
            if self.on_speech_start:
                self.on_speech_start()
            
            logger.debug("Speech started")
        
        elif not is_speaking and self._was_speaking:
            if self.on_speech_end:
                self.on_speech_end()
            
            logger.debug("Speech ended")
        
        if self.on_interrupt and self._is_speaking and self._was_speaking:
            if self._is_tts_active():
                self._handle_interrupt(audio_chunk)

    def _is_tts_active(self) -> bool:
        """Check if TTS is currently playing."""
        return getattr(self, '_tts_active', False)

    def set_tts_active(self, active: bool):
        """Mark TTS as active/inactive for interrupt detection."""
        self._tts_active = active

    def _handle_interrupt(self, audio_chunk: np.ndarray):
        """Handle user interruption during TTS."""
        logger.info("Interrupt detected during TTS")
        
        self._interrupt_buffer.append({
            "timestamp": datetime.now().isoformat(),
            "audio_preview": "interrupted_speech"
        })
        
        if self.on_interrupt:
            self.on_interrupt(audio_chunk)

    def append_interrupt_to_context(self, transcript: str):
        """Append interruption to the active context window."""
        interruption_entry = {
            "type": "interruption",
            "content": transcript,
            "timestamp": datetime.now().isoformat()
        }
        
        self._context_window.append(interruption_entry)
        
        if len(self._context_window) > self._max_context_items:
            self._context_window = self._context_window[-self._max_context_items:]
        
        logger.info(f"Appended interruption to context: {transcript[:50]}...")

    def get_context_window(self) -> list:
        """Get the current context window including interruptions."""
        return self._context_window.copy()

    def clear_context(self):
        """Clear the context window."""
        self._context_window.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get acoustic awareness statistics."""
        return {
            "running": self._running,
            "is_speaking": self._is_speaking,
            "vad_threshold": self.vad_threshold,
            "buffer_size_ms": self.buffer_size_ms,
            "context_items": len(self._context_window),
            "interrupt_count": len(self._interrupt_buffer)
        }


class AudioBuffer:
    """Rolling audio buffer for VAD analysis."""

    def __init__(self, max_duration_sec: float = 5.0, sample_rate: int = 16000):
        self.max_duration_sec = max_duration_sec
        self.sample_rate = sample_rate
        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

    def append(self, audio_data: np.ndarray):
        """Append audio data to buffer."""
        with self._lock:
            self._buffer = np.concatenate([self._buffer, audio_data])
            
            max_samples = int(self.max_duration_sec * self.sample_rate)
            if len(self._buffer) > max_samples:
                self._buffer = self._buffer[-max_samples:]

    def get_recent(self, duration_ms: int) -> np.ndarray:
        """Get recent audio segment."""
        with self._lock:
            samples = int(duration_ms * self.sample_rate / 1000)
            if len(self._buffer) >= samples:
                return self._buffer[-samples:].copy()
            return self._buffer.copy()

    def get_all(self) -> np.ndarray:
        """Get all buffered audio."""
        with self._lock:
            return self._buffer.copy()

    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self._buffer = np.zeros(0, dtype=np.float32)


class TTSInterruptHandler:
    """Handles TTS interruption during playback."""

    def __init__(self, acoustic_awareness: AcousticAwareness):
        self.acoustic = acoustic_awareness
        self._interrupted = False

    def start_tts(self):
        """Mark TTS as starting."""
        self.acoustic.set_tts_active(True)
        self._interrupted = False

    def stop_tts(self):
        """Mark TTS as stopped."""
        self.acoustic.set_tts_active(False)

    def was_interrupted(self) -> bool:
        """Check if TTS was interrupted."""
        return self._interrupted

    def handle_interrupt(self, audio_chunk: np.ndarray):
        """Handle the interrupt - halt playback."""
        self._interrupted = True
        self.stop_tts()
        
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.stop()
        except Exception as e:
            logger.debug(f"Could not stop pygame mixer: {e}")
        
        logger.info("TTS playback halted due to interruption")


_acoustic_awareness: Optional[AcousticAwareness] = None


def get_acoustic_awareness() -> AcousticAwareness:
    global _acoustic_awareness
    if _acoustic_awareness is None:
        _acoustic_awareness = AcousticAwareness()
    return _acoustic_awareness


def start_acoustic_awareness():
    awareness = get_acoustic_awareness()
    awareness.start()
    return awareness


def stop_acoustic_awareness():
    global _acoustic_awareness
    if _acoustic_awareness:
        _acoustic_awareness.stop()