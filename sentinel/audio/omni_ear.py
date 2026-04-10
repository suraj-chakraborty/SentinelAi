"""
sentinel/audio/omni_ear.py
───────────────────────────
System-Wide Audio Intelligence - The "Omni-Ear".

Captures all system audio (speakers) in real-time:
- WASAPI Loopback capture
- Continuous Whisper transcription
- 30-day searchable transcript
- Historical search of heard audio
"""

import os
import time
import logging
import threading
import queue
import json
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from sentinel.app.config import APPDATA_DIR

logger = logging.getLogger("OmniEar")

OMNI_EAR_DIR = os.path.join(APPDATA_DIR, "audio", "omni_ear")
os.makedirs(OMNI_EAR_DIR, exist_ok=True)

TRANSCRIPTS_DIR = os.path.join(OMNI_EAR_DIR, "transcripts")
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

TRANSCRIPT_INDEX_FILE = os.path.join(OMNI_EAR_DIR, "transcript_index.json")


@dataclass
class AudioSegment:
    """Transcribed audio segment."""
    segment_id: str
    timestamp: datetime
    duration_sec: float
    text: str
    source: str
    confidence: float = 0.0


class WASAPILoopback:
    """Capture system audio via Windows WASAPI loopback."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream = None
        self._pyaudio = None
        self._init_audio()

    def _init_audio(self):
        """Initialize PyAudio with WASAPI loopback."""
        try:
            import pyaudio
            
            self._pyaudio = pyaudio.PyAudio()
            
            for i in range(self._pyaudio.get_device_count()):
                device_info = self._pyaudio.get_device_info_by_index(i)
                if "Stereo Mix" in device_info.get("name", "") or "Loopback" in device_info.get("name", ""):
                    logger.info(f"Found loopback device: {device_info['name']}")
                    self._loopback_device = i
                    return
            
            logger.warning("No loopback device found, using default")
            self._loopback_device = None
        
        except Exception as e:
            logger.error(f"Failed to init PyAudio: {e}")
            self._pyaudio = None

    def start_capture(self, callback: Callable[[np.ndarray], None]) -> bool:
        """Start capturing system audio."""
        if not self._pyaudio:
            return False
        
        try:
            def audio_callback(in_data, frame_count, time_info, status):
                audio_data = np.frombuffer(in_data, dtype=np.int16)
                callback(audio_data.astype(np.float32) / 32768.0)
                return (in_data, pyaudio.paContinue)
            
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self._loopback_device,
                stream_callback=audio_callback
            )
            
            self._stream.start_stream()
            logger.info("System audio capture started")
            return True
        
        except Exception as e:
            logger.error(f"Failed to start capture: {e}")
            return False

    def stop_capture(self):
        """Stop capturing system audio."""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        logger.info("System audio capture stopped")


class WhisperTranscriber:
    """Transcribe audio using Whisper."""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None
        self._init_model()

    def _init_model(self):
        """Initialize Whisper model."""
        try:
            import whisper
            self._model = whisper.load_model(self.model_name)
            logger.info(f"Whisper model '{self.model_name}' loaded")
        except Exception as e:
            logger.warning(f"Whisper not available: {e}")

    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio segment."""
        if self._model is None:
            return None
        
        try:
            result = self._model.transcribe(
                audio,
                language="en",
                fp16=False,
                initial_prompt="This is system audio from a computer."
            )
            return result.get("text", "").strip()
        except Exception as e:
            logger.debug(f"Transcription error: {e}")
            return None


class TranscriptStore:
    """Store and search audio transcripts."""

    def __init__(self, max_segments: int = 10000, retention_days: int = 30):
        self.max_segments = max_segments
        self.retention_days = retention_days
        self._segments: List[AudioSegment] = []
        self._load_index()

    def add_segment(self, segment: AudioSegment):
        """Add a transcribed segment."""
        self._segments.append(segment)
        
        if len(self._segments) > self.max_segments:
            self._segments = self._segments[-self.max_segments:]
        
        self._save_index()

    def search(self, query: str, days_back: Optional[int] = None) -> List[AudioSegment]:
        """Search transcripts."""
        results = []
        
        query_lower = query.lower()
        
        for segment in self._segments:
            if query_lower in segment.text.lower():
                if days_back:
                    age = (datetime.now() - segment.timestamp).days
                    if age > days_back:
                        continue
                results.append(segment)
        
        return results[-20:]

    def get_recent(self, hours: int = 24) -> List[AudioSegment]:
        """Get recent transcripts."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [s for s in self._segments if s.timestamp > cutoff]

    def _save_index(self):
        """Save transcript index."""
        try:
            data = [
                {
                    "segment_id": s.segment_id,
                    "timestamp": s.timestamp.isoformat(),
                    "duration_sec": s.duration_sec,
                    "text": s.text,
                    "source": s.source,
                    "confidence": s.confidence
                }
                for s in self._segments
            ]
            
            with open(TRANSCRIPT_INDEX_FILE, "w") as f:
                json.dump(data, f)
        
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def _load_index(self):
        """Load transcript index."""
        if not os.path.exists(TRANSCRIPT_INDEX_FILE):
            return
        
        try:
            with open(TRANSCRIPT_INDEX_FILE, "r") as f:
                data = json.load(f)
            
            self._segments = [
                AudioSegment(
                    segment_id=d["segment_id"],
                    timestamp=datetime.fromisoformat(d["timestamp"]),
                    duration_sec=d["duration_sec"],
                    text=d["text"],
                    source=d["source"],
                    confidence=d.get("confidence", 0.0)
                )
                for d in data
            ]
            
            logger.info(f"Loaded {len(self._segments)} transcript segments")
        
        except Exception as e:
            logger.error(f"Failed to load index: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get transcript stats."""
        return {
            "total_segments": len(self._segments),
            "retention_days": self.retention_days,
            "earliest": self._segments[0].timestamp.isoformat() if self._segments else None,
            "latest": self._segments[-1].timestamp.isoformat() if self._segments else None
        }


class OmniEar:
    """
    System-Wide Audio Intelligence - The "Omni-Ear".
    
    Features:
    - WASAPI loopback capture
    - Real-time Whisper transcription
    - 30-day searchable transcript
    - Historical search of system audio
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        whisper_model: str = "base",
        buffer_size_sec: int = 10,
        on_transcript: Optional[Callable] = None
    ):
        self.sample_rate = sample_rate
        self.whisper_model = whisper_model
        self.buffer_size_sec = buffer_size_sec
        self.on_transcript = on_transcript
        
        self._loopback = WASAPILoopback(sample_rate=sample_rate)
        self._transcriber = WhisperTranscriber(model_name=whisper_model)
        self._transcript_store = TranscriptStore()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self._audio_buffer: List[np.ndarray] = []
        self._segment_count = 0

    def start(self):
        """Start the Omni-Ear."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="OmniEar")
        self._thread.start()
        
        logger.info(f"Omni-Ear started (Whisper: {self.whisper_model})")

    def stop(self):
        """Stop the Omni-Ear."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._loopback.stop_capture()
        logger.info("Omni-Ear stopped")

    def _capture_loop(self):
        """Main audio capture and transcription loop."""
        self._loopback.start_capture(self._feed_audio)
        
        while self._running:
            time.sleep(1)
            
            if len(self._audio_buffer) >= self.buffer_size_sec:
                self._process_buffer()

    def _feed_audio(self, audio_chunk: np.ndarray):
        """Feed audio chunk to buffer."""
        self._audio_buffer.append(audio_chunk)
        
        max_chunks = self.buffer_size_sec * self.sample_rate // 16000
        if len(self._audio_buffer) > max_chunks:
            self._audio_buffer = self._audio_buffer[-max_chunks:]

    def _process_buffer(self):
        """Process accumulated audio buffer."""
        if not self._audio_buffer:
            return
        
        try:
            audio = np.concatenate(self._audio_buffer)
            self._audio_buffer = []
            
            if len(audio) < self.sample_rate:
                return
            
            text = self._transcriber.transcribe(audio)
            
            if text and len(text) > 10:
                segment = AudioSegment(
                    segment_id=f"seg_{self._segment_count}_{int(time.time())}",
                    timestamp=datetime.now(),
                    duration_sec=len(audio) / self.sample_rate,
                    text=text,
                    source="system_audio"
                )
                
                self._transcript_store.add_segment(segment)
                self._segment_count += 1
                
                if self.on_transcript:
                    self.on_transcript(text)
                
                logger.debug(f"Transcribed: {text[:50]}...")
        
        except Exception as e:
            logger.error(f"Buffer processing error: {e}")

    def search(self, query: str, days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search transcript history."""
        results = self._transcript_store.search(query, days_back)
        
        return [
            {
                "segment_id": r.segment_id,
                "timestamp": r.timestamp.isoformat(),
                "text": r.text,
                "source": r.source,
                "duration_sec": r.duration_sec
            }
            for r in results
        ]

    def get_recent(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent transcripts."""
        recent = self._transcript_store.get_recent(hours)
        
        return [
            {
                "segment_id": r.segment_id,
                "timestamp": r.timestamp.isoformat(),
                "text": r.text,
                "source": r.source
            }
            for r in recent
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get Omni-Ear statistics."""
        store_stats = self._transcript_store.get_stats()
        return {
            "running": self._running,
            "whisper_model": self.whisper_model,
            "buffer_sec": self.buffer_size_sec,
            **store_stats
        }


_omni_ear: Optional[OmniEar] = None


def get_omni_ear() -> OmniEar:
    global _omni_ear
    if _omni_ear is None:
        _omni_ear = OmniEar()
    return _omni_ear


def start_omni_ear(whisper_model: str = "base") -> OmniEar:
    ear = get_omni_ear()
    ear.whisper_model = whisper_model
    ear.start()
    return ear


def stop_omni_ear():
    global _omni_ear
    if _omni_ear:
        ear.stop()


def search_audio_history(query: str, days_back: Optional[int] = None) -> List[Dict[str, Any]]:
    """Search audio transcript history."""
    ear = get_omni_ear()
    return ear.search(query, days_back)