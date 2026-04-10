"""
sentinel/voice/emotional_tts.py
───────────────────────────────
Biomimetic "Breathing" & Emotional Synthesis Module.

Makes TTS sound indistinguishable from a living entity:
- Action tags for breathing/hesitations
- Human-like cognitive delays
- Emotional adaptation to user state
- Stress detection from typing patterns
"""

import os
import time
import logging
import re
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime
import threading
import queue

import numpy as np

from sentinel.app.config import APPDATA_DIR

logger = logging.getLogger("EmotionalTTS")

EMOTIONAL_TTS_DIR = os.path.join(APPDATA_DIR, "audio", "emotional")
os.makedirs(EMOTIONAL_TTS_DIR, exist_ok=True)

BREATH_SAMPLES_DIR = os.path.join(EMOTIONAL_TTS_DIR, "breath_samples")
os.makedirs(BREATH_SAMPLES_DIR, exist_ok=True)


class ActionTag:
    """Represents an action tag in TTS text."""
    TAG_PATTERNS = {
        r"\[sigh\]": "sigh",
        r"\[short_breath\]": "short_breath",
        r"\[long_breath\]": "long_breath",
        r"\[cough\]": "cough",
        r"\[umm\]": "umm",
        r"\[well\]": "well",
        r"\[uhh\]": "uhh",
        r"\[pause\]": "pause",
        r"\[short_pause\]": "short_pause",
        r"\[long_pause\]": "long_pause",
        r"\[think\]": "think",
        r"\[hmm\]": "hmm",
        r"\[laugh\]": "laugh",
        r"\[soft_laugh\]": "soft_laugh"
    }


@dataclass
class TTSChunk:
    """TTS chunk to be spoken."""
    text: str
    is_action: bool
    action_type: Optional[str] = None
    duration_ms: int = 0


class BreathSampleGenerator:
    """Generate breath samples programmatically."""
    
    def __init__(self):
        self._samples = {}
    
    def generate_sigh(self) -> np.ndarray:
        """Generate sigh sound."""
        duration = 0.8
        sample_rate = 24000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        freq = 150 + 50 * np.sin(2 * np.pi * 2 * t)
        envelope = np.exp(-3 * t) * (1 - np.exp(-5 * t))
        
        audio = 0.3 * envelope * np.sin(2 * np.pi * freq * t)
        audio += 0.1 * envelope * np.sin(2 * np.pi * freq * 2 * t)
        
        return audio.astype(np.float32)
    
    def generate_short_breath(self) -> np.ndarray:
        """Generate short breath sound."""
        duration = 0.3
        sample_rate = 24000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        envelope = np.exp(-10 * t) * (1 - np.exp(-20 * t))
        
        noise = np.random.randn(len(t)) * 0.05
        audio = envelope * noise
        
        return audio.astype(np.float32)
    
    def generate_long_breath(self) -> np.ndarray:
        """Generate long breath sound."""
        duration = 0.6
        sample_rate = 24000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        envelope = np.exp(-5 * t) * (1 - np.exp(-15 * t))
        
        noise = np.random.randn(len(t)) * 0.08
        audio = envelope * noise
        
        return audio.astype(np.float32)
    
    def generate_umm(self) -> np.ndarray:
        """Generate 'umm' hesitation sound."""
        duration = 0.4
        sample_rate = 24000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        freq = 200 + 20 * np.sin(2 * np.pi * 3 * t)
        envelope = np.exp(-8 * t)
        
        audio = 0.2 * envelope * np.sin(2 * np.pi * freq * t)
        
        return audio.astype(np.float32)
    
    def generate_well(self) -> np.ndarray:
        """Generate 'well' hesitation sound."""
        duration = 0.35
        sample_rate = 24000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        freq = 180 + 30 * np.sin(2 * np.pi * 4 * t)
        envelope = np.exp(-10 * t)
        
        audio = 0.2 * envelope * np.sin(2 * np.pi * freq * t)
        
        return audio.astype(np.float32)
    
    def generate_hmm(self) -> np.ndarray:
        """Generate 'hmm' thinking sound."""
        duration = 0.5
        sample_rate = 24000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        freq = 120
        envelope = np.exp(-6 * t)
        
        audio = 0.25 * envelope * np.sin(2 * np.pi * freq * t)
        audio += 0.1 * envelope * np.sin(2 * np.pi * freq * 2 * t)
        
        return audio.astype(np.float32)
    
    def generate_pause(self, duration_ms: int = 500) -> np.ndarray:
        """Generate silence/pause."""
        samples = int(24000 * duration_ms / 1000)
        return np.zeros(samples, dtype=np.float32)
    
    def get_sample(self, action_type: str) -> np.ndarray:
        """Get breath/hesitation sample."""
        generators = {
            "sigh": self.generate_sigh,
            "short_breath": self.generate_short_breath,
            "long_breath": self.generate_long_breath,
            "umm": self.generate_umm,
            "well": self.generate_well,
            "hmm": self.generate_hmm,
            "pause": lambda: self.generate_pause(500),
            "short_pause": lambda: self.generate_pause(200),
            "long_pause": lambda: self.generate_pause(800)
        }
        
        if action_type in generators:
            if action_type not in self._samples:
                self._samples[action_type] = generators[action_type]()
            return self._samples[action_type]
        
        return np.zeros(1000, dtype=np.float32)


class EmotionalTTSParser:
    """Parse text for action tags."""
    
    def __init__(self):
        self._pattern = re.compile("|".join(ActionTag.TAG_PATTERNS.keys()))
    
    def parse(self, text: str) -> List[TTSChunk]:
        """Parse text into TTS chunks."""
        chunks = []
        last_end = 0
        
        for match in self._pattern.finditer(text):
            if match.start() > last_end:
                chunks.append(TTSChunk(
                    text=text[last_end:match.start()],
                    is_action=False
                ))
            
            action_type = ActionTag.TAG_PATTERNS.get(match.group(), "pause")
            
            durations = {
                "sigh": 800,
                "short_breath": 300,
                "long_breath": 600,
                "umm": 400,
                "well": 350,
                "hmm": 500,
                "pause": 500,
                "short_pause": 200,
                "long_pause": 800,
                "cough": 600,
                "think": 700,
                "laugh": 400,
                "soft_laugh": 300
            }
            
            chunks.append(TTSChunk(
                text=match.group(),
                is_action=True,
                action_type=action_type,
                duration_ms=durations.get(action_type, 300)
            ))
            
            last_end = match.end()
        
        if last_end < len(text):
            chunks.append(TTSChunk(
                text=text[last_end:],
                is_action=False
            ))
        
        return [c for c in chunks if c.text.strip()]


class UserStateAnalyzer:
    """Analyze user state for emotional adaptation."""
    
    def __init__(self):
        self._typing_samples: List[float] = []
        self._last_typing_time = None
        self._sample_window_sec = 30
    
    def record_keystroke(self, timestamp: float):
        """Record a keystroke for typing pattern analysis."""
        if self._last_typing_time:
            interval = timestamp - self._last_typing_time
            if interval < 1.0:
                self._typing_samples.append(interval)
        
        self._last_typing_time = timestamp
        
        cutoff = timestamp - self._sample_window_sec
        self._typing_samples = [s for s in self._typing_samples if s > cutoff]
    
    def analyze_stress_level(self) -> str:
        """Analyze current stress level from typing."""
        if len(self._typing_samples) < 5:
            return "normal"
        
        avg_interval = sum(self._typing_samples) / len(self._typing_samples)
        
        if avg_interval < 0.08:
            return "high_stress"
        elif avg_interval < 0.15:
            return "moderate"
        else:
            return "calm"
    
    def get_voice_adjustment(self) -> Dict[str, Any]:
        """Get voice adjustment based on user state."""
        stress = self.analyze_stress_level()
        
        adjustments = {
            "high_stress": {
                "pitch_shift": -0.15,
                "speed_multiplier": 0.85,
                "volume_db": -3,
                "formality": "formal",
                "tone": "calm"
            },
            "moderate": {
                "pitch_shift": 0,
                "speed_multiplier": 0.95,
                "volume_db": 0,
                "formality": "neutral",
                "tone": "neutral"
            },
            "calm": {
                "pitch_shift": 0.05,
                "speed_multiplier": 1.0,
                "volume_db": 0,
                "formality": "casual",
                "tone": "friendly"
            },
            "normal": {
                "pitch_shift": 0,
                "speed_multiplier": 1.0,
                "volume_db": 0,
                "formality": "neutral",
                "tone": "neutral"
            }
        }
        
        return adjustments.get(stress, adjustments["normal"])


class EmotionalTTS:
    """
    Biomimetic "Breathing" & Emotional Synthesis TTS.
    
    Features:
    - Action tags for breathing/hesitations
    - Human-like cognitive delays
    - Emotional adaptation to user stress
    - Personalized voice adjustments
    """

    def __init__(
        self,
        base_tts_callback: Optional[Callable] = None,
        on_generate: Optional[Callable] = None
    ):
        self.base_tts_callback = base_tts_callback
        self.on_generate = on_generate
        
        self._parser = EmotionalTTSParser()
        self._breath_gen = BreathSampleGenerator()
        self._user_analyzer = UserStateAnalyzer()
        
        self._current_adjustment = {
            "pitch_shift": 0,
            "speed_multiplier": 1.0,
            "volume_db": 0
        }
        
        self._enabled = True
        self._inject_probability = 0.3

    def process_text(self, text: str, add_auto_tags: bool = True) -> str:
        """Process text and optionally add automatic action tags."""
        if add_auto_tags and self._enabled:
            text = self._add_cognitive_delays(text)
        
        return text
    
    def _add_cognitive_delays(self, text: str) -> str:
        """Add automatic cognitive delay tags."""
        words = text.split()
        
        if len(words) < 5:
            return text
        
        for i, word in enumerate(words):
            if np.random.random() < self._inject_probability:
                if i > 0:
                    tags = ["[short_pause]", "[hmm]", "[well]", "[umm]"]
                    tag = np.random.choice(tags)
                    
                    words[i] = f"{tag} {word}"
        
        return " ".join(words)

    def speak(self, text: str, add_auto_tags: bool = True) -> bool:
        """Speak text with emotional synthesis."""
        processed_text = self.process_text(text, add_auto_tags)
        chunks = self._parser.parse(processed_text)
        
        try:
            for chunk in chunks:
                if chunk.is_action:
                    audio = self._breath_gen.get_sample(chunk.action_type)
                    
                    if self.on_generate:
                        self.on_generate(audio)
                    
                    duration = chunk.duration_ms / 1000
                    time.sleep(duration)
                else:
                    if self.base_tts_callback:
                        adjusted_text = self._adjust_text(chunk.text)
                        self.base_tts_callback(adjusted_text)
        
        except Exception as e:
            logger.error(f"Emotional TTS error: {e}")
            return False
        
        return True

    def _adjust_text(self, text: str) -> str:
        """Adjust text based on current emotional state."""
        return text

    def set_voice_adjustment(self, adjustment: Dict[str, Any]):
        """Set voice adjustment parameters."""
        self._current_adjustment = adjustment
        logger.info(f"Voice adjustment set: {adjustment}")

    def adapt_to_user_state(self):
        """Adapt TTS to current user state."""
        adjustment = self._user_analyzer.get_voice_adjustment()
        self.set_voice_adjustment(adjustment)
        
        logger.info(f"Adapted to user state: {adjustment.get('tone')}")

    def record_typing(self, timestamp: float):
        """Record typing for stress analysis."""
        self._user_analyzer.record_keystroke(timestamp)

    def enable(self):
        """Enable emotional TTS."""
        self._enabled = True

    def disable(self):
        """Disable emotional TTS."""
        self._enabled = False

    def get_stats(self) -> Dict[str, Any]:
        """Get emotional TTS statistics."""
        return {
            "enabled": self._enabled,
            "inject_probability": self._inject_probability,
            "current_adjustment": self._current_adjustment,
            "stress_level": self._user_analyzer.analyze_stress_level()
        }


_emotional_tts: Optional[EmotionalTTS] = None


def get_emotional_tts() -> EmotionalTTS:
    global _emotional_tts
    if _emotional_tts is None:
        _emotional_tts = EmotionalTTS()
    return _emotional_tts


def speak_with_emotion(text: str, auto_tags: bool = True) -> bool:
    """Speak text with emotional synthesis."""
    tts = get_emotional_tts()
    return tts.speak(text, auto_tags)


def adapt_tts_to_user():
    """Adapt TTS to current user state."""
    tts = get_emotional_tts()
    tts.adapt_to_user_state()


def record_user_typing(timestamp: float):
    """Record typing for stress analysis."""
    tts = get_emotional_tts()
    tts.record_typing(timestamp)