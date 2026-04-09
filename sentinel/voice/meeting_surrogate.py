"""
sentinel/voice/meeting_surrogate.py
───────────────────────────────────
Social Proxy - Meeting Surrogate Module.

Allows Sentinel to attend digital meetings (Zoom/Teams) on your behalf:
- Route TTS to virtual microphone (VB-Audio)
- Transcribe meeting audio
- Respond as meeting surrogate
- Generate meeting summaries
"""

import os
import time
import logging
import threading
import queue
import json
import uuid
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import tempfile

import numpy as np

from sentinel.app.config import APPDATA_DIR

logger = logging.getLogger("MeetingSurrogate")

MEETING_DIR = os.path.join(APPDATA_DIR, "meetings")
os.makedirs(MEETING_DIR, exist_ok=True)


@dataclass
class MeetingTranscript:
    """Meeting transcript entry."""
    speaker: str
    text: str
    timestamp: datetime
    is_sentinel: bool = False


@dataclass
class MeetingSummary:
    """Meeting summary with action items."""
    title: str
    date: datetime
    duration_minutes: int
    participants: List[str]
    transcript: List[MeetingTranscript]
    action_items: List[str]
    decisions: List[str]


class AudioRouter:
    """Route audio between system, TTS, and virtual devices."""

    def __init__(self):
        self._virtual_cable_available = False
        self._check_virtual_cable()

    def _check_virtual_cable(self):
        """Check if VB-Audio Virtual Cable is available."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            for dev in devices:
                if "Virtual" in dev.get("name", "") or "CABLE" in dev.get("name", ""):
                    self._virtual_cable_available = True
                    logger.info("Virtual audio cable detected")
                    return
        except Exception:
            pass
        
        logger.warning("Virtual audio cable not available")
        self._virtual_cable_available = False

    def is_available(self) -> bool:
        """Check if audio routing is available."""
        return self._virtual_cable_available

    def route_tts_to_virtual(self, audio_data: np.ndarray) -> bool:
        """Route TTS output to virtual microphone."""
        if not self._virtual_cable_available:
            logger.warning("Cannot route - virtual cable not available")
            return False
        
        try:
            import sounddevice as sd
            sd.play(audio_data, samplerate=16000)
            return True
        except Exception as e:
            logger.error(f"Failed to route TTS: {e}")
            return False

    def capture_system_audio(self, duration_sec: float = 1.0) -> Optional[np.ndarray]:
        """Capture system audio (meeting audio)."""
        if not self._virtual_cable_available:
            return None
        
        try:
            import sounddevice as sd
            audio = sd.rec(int(duration_sec * 16000), samplerate=16000, channels=1)
            return audio
        except Exception as e:
            logger.debug(f"Audio capture failed: {e}")
            return None


class MeetingTranscriber:
    """Transcribe meeting audio using Whisper/AssemblyAI."""

    def __init__(self):
        self._model = None
        self._init_model()

    def _init_model(self):
        """Initialize Whisper model."""
        try:
            import whisper
            self._model = whisper.load_model("base")
            logger.info("Whisper model loaded")
        except Exception as e:
            logger.warning(f"Whisper not available: {e}")

    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio segment."""
        if self._model is None:
            return None
        
        try:
            result = self._model.transcribe(audio, language="en")
            return result.get("text", "").strip()
        except Exception as e:
            logger.debug(f"Transcription failed: {e}")
            return None

    def transcribe_file(self, filepath: str) -> Optional[str]:
        """Transcribe audio file."""
        if self._model is None:
            return None
        
        try:
            result = self._model.transcribe(filepath)
            return result.get("text", "").strip()
        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            return None


class MeetingSurrogate:
    """
    Social Proxy - Meeting Surrogate Agent.
    
    Features:
    - Virtual audio routing (VB-Audio)
    - Real-time transcription
    - Meeting persona responses
    - Meeting summary generation
    """

    def __init__(
        self,
        user_name: str = "User",
        on_response: Optional[Callable] = None
    ):
        self.user_name = user_name
        
        self._audio_router = AudioRouter()
        self._transcriber = MeetingTranscriber()
        self.on_response = on_response
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self._is_in_meeting = False
        self._meeting_start_time: Optional[datetime] = None
        self._transcript: List[MeetingTranscript] = []
        self._participants: List[str] = []
        self._action_items: List[str] = []
        
        self._persona = f"""You are Sentinel, an AI assistant representing {self.user_name}.
You are in a meeting. Provide concise answers and take notes.
Do not dominate the conversation - respond only when addressed or relevant."""

    def start_meeting(self, meeting_id: Optional[str] = None) -> bool:
        """Start attending a meeting."""
        if self._is_in_meeting:
            logger.warning("Already in a meeting")
            return False
        
        self._is_in_meeting = True
        self._meeting_start_time = datetime.now()
        self._transcript = []
        self._participants = []
        self._action_items = []
        self._meeting_id = meeting_id or str(uuid.uuid4())[:8]
        
        self._running = True
        self._thread = threading.Thread(target=self._meeting_loop, daemon=True, name="MeetingSurrogate")
        self._thread.start()
        
        logger.info(f"Meeting started: {self._meeting_id}")
        
        try:
            from sentinel.voice.tts import speak
            speak(f"I have joined the meeting. I'll take notes and respond on your behalf.")
        except Exception:
            pass
        
        return True

    def end_meeting(self) -> Optional[MeetingSummary]:
        """End the meeting and generate summary."""
        if not self._is_in_meeting:
            return None
        
        self._running = False
        self._is_in_meeting = False
        
        duration = (datetime.now() - self._meeting_start_time).total_seconds() / 60
        
        summary = MeetingSummary(
            title=f"Meeting {self._meeting_id}",
            date=self._meeting_start_time,
            duration_minutes=int(duration),
            participants=self._participants,
            transcript=self._transcript,
            action_items=self._action_items,
            decisions=[]
        )
        
        self._save_summary(summary)
        
        logger.info(f"Meeting ended: {self._meeting_id}, duration: {duration:.1f}min")
        
        try:
            from sentinel.voice.tts import speak
            speak(f"Meeting ended. I recorded {len(self._transcript)} entries and {len(self._action_items)} action items.")
        except Exception:
            pass
        
        return summary

    def _meeting_loop(self):
        """Main meeting monitoring loop."""
        audio_buffer = []
        
        while self._running:
            try:
                audio = self._audio_router.capture_system_audio(duration_sec=1.0)
                
                if audio is not None and len(audio) > 0:
                    audio_buffer.append(audio)
                    
                    if len(audio_buffer) >= 5:
                        combined = np.concatenate(audio_buffer[-5:])
                        text = self._transcriber.transcribe(combined)
                        
                        if text:
                            self._process_transcript(text)
                        
                        audio_buffer = audio_buffer[-2:]
                
                time.sleep(0.5)
            
            except Exception as e:
                logger.error(f"Meeting loop error: {e}")
                time.sleep(1)

    def _process_transcript(self, text: str):
        """Process incoming transcript text."""
        lower_text = text.lower()
        
        if self.user_name.lower() in lower_text or f"hey {self.user_name.split()[0]}".lower() in lower_text:
            if self._should_respond(text):
                self._respond_in_meeting(text)
        
        if "action:" in lower_text or "todo:" in lower_text:
            self._action_items.append(text)
        
        entry = MeetingTranscript(
            speaker="Unknown",
            text=text,
            timestamp=datetime.now(),
            is_sentinel=False
        )
        
        self._transcript.append(entry)

    def _should_respond(self, text: str) -> bool:
        """Determine if Sentinel should respond."""
        respond_triggers = [
            f"where is {self.user_name.lower()}",
            f"is {self.user_name.lower()} here",
            f"{self.user_name.lower()} can you",
            "sentinel can you",
            "assistant can you"
        ]
        
        return any(trigger in text.lower() for trigger in respond_triggers)

    def _respond_in_meeting(self, context: str):
        """Generate and deliver response in meeting."""
        if not self.on_response:
            return
        
        try:
            from sentinel.core.orchestrator import get_orchestrator
            orc = get_orchestrator()
            
            if orc and hasattr(orc, "_safe_llm_call"):
                prompt = f"""{self._persona}

Someone asked about {self.user_name} in the meeting.
Context: {context}

Provide a brief, polite response (1-2 sentences)."""
                
                response = orc._safe_llm_call(prompt)
                
                if response:
                    self._speak_in_meeting(response)
                    
                    entry = MeetingTranscript(
                        speaker="Sentinel",
                        text=response,
                        timestamp=datetime.now(),
                        is_sentinel=True
                    )
                    self._transcript.append(entry)
        
        except Exception as e:
            logger.error(f"Response generation failed: {e}")

    def _speak_in_meeting(self, text: str):
        """Speak response through virtual microphone."""
        try:
            from sentinel.voice.tts import generate_audio
            audio_data = generate_audio(text)
            
            if audio_data is not None:
                self._audio_router.route_tts_to_virtual(audio_data)
        
        except Exception as e:
            logger.error(f"Meeting speech failed: {e}")

    def _save_summary(self, summary: MeetingSummary):
        """Save meeting summary to file."""
        filepath = os.path.join(
            MEETING_DIR,
            f"meeting_{summary.title}_{summary.date.strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        try:
            data = {
                "title": summary.title,
                "date": summary.date.isoformat(),
                "duration_minutes": summary.duration_minutes,
                "participants": summary.participants,
                "transcript": [
                    {
                        "speaker": t.speaker,
                        "text": t.text,
                        "timestamp": t.timestamp.isoformat(),
                        "is_sentinel": t.is_sentinel
                    }
                    for t in summary.transcript
                ],
                "action_items": summary.action_items,
                "decisions": summary.decisions
            }
            
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Meeting summary saved: {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")

    def get_transcript(self) -> List[MeetingTranscript]:
        """Get current meeting transcript."""
        return self._transcript.copy()

    def get_action_items(self) -> List[str]:
        """Get meeting action items."""
        return self._action_items.copy()

    def is_in_meeting(self) -> bool:
        """Check if currently in a meeting."""
        return self._is_in_meeting

    def get_stats(self) -> Dict[str, Any]:
        """Get meeting surrogate statistics."""
        return {
            "in_meeting": self._is_in_meeting,
            "meeting_id": self._meeting_id if self._is_in_meeting else None,
            "transcript_entries": len(self._transcript),
            "action_items": len(self._action_items),
            "virtual_audio_available": self._audio_router.is_available()
        }


_meeting_surrogate: Optional[MeetingSurrogate] = None


def get_meeting_surrogate() -> MeetingSurrogate:
    global _meeting_surrogate
    if _meeting_surrogate is None:
        _meeting_surrogate = MeetingSurrogate()
    return _meeting_surrogate


def start_meeting(user_name: str = "User") -> bool:
    surrogate = get_meeting_surrogate()
    surrogate.user_name = user_name
    return surrogate.start_meeting()


def end_meeting() -> Optional[MeetingSummary]:
    surrogate = get_meeting_surrogate()
    return surrogate.end_meeting()