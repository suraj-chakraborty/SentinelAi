"""
sentinel/ar/holographic_avatar.py
──────────────────────────────────
Holographic 3D Floating Avatar & Real-Time Lip-Sync Engine.

Provides:
  1. Desktop-overlay transparent avatar (holographic orb or stylized companion).
  2. Phoneme-driven real-time lip-sync mapped from audio stream.
  3. Emotional visual states: puffed cheeks (cute anger), blush (flirty),
     cyan pulse (focused), warm glow (empathic).
  4. Cursor/gaze-aware eye tracking.
  5. GPU-accelerated particle system for ambient holographic effects.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("HolographicAvatar")


# ── Phoneme-to-Viseme Map ────────────────────────────────────────────────────

class Viseme(Enum):
    """Standard viseme categories for mouth shape animation."""
    SILENCE    = "silence"       # Mouth closed
    AA         = "aa"            # Open jaw (father)
    EE         = "ee"            # Spread lips (see)
    OH         = "oh"            # Rounded lips (go)
    OO         = "oo"            # Pursed lips (too)
    MM         = "mm"            # Closed lips (mama)
    TH         = "th"            # Tongue between teeth
    FF         = "ff"            # Teeth on lip (fine)
    SS         = "ss"            # Teeth together (see)
    SH         = "sh"            # Rounded slightly (shoe)
    WIDE_OPEN  = "wide_open"     # Exclamation / emphasis


PHONEME_TO_VISEME = {
    "AA": Viseme.AA, "AE": Viseme.AA, "AH": Viseme.AA, "AO": Viseme.OH,
    "AW": Viseme.OH, "AY": Viseme.AA,
    "B": Viseme.MM, "CH": Viseme.SH,
    "D": Viseme.TH, "DH": Viseme.TH,
    "EH": Viseme.EE, "ER": Viseme.OH, "EY": Viseme.EE,
    "F": Viseme.FF, "G": Viseme.AA,
    "HH": Viseme.AA,
    "IH": Viseme.EE, "IY": Viseme.EE,
    "JH": Viseme.SH, "K": Viseme.AA,
    "L": Viseme.TH, "M": Viseme.MM, "N": Viseme.TH, "NG": Viseme.AA,
    "OW": Viseme.OH, "OY": Viseme.OH,
    "P": Viseme.MM, "R": Viseme.OH,
    "S": Viseme.SS, "SH": Viseme.SH,
    "T": Viseme.TH, "TH": Viseme.TH,
    "UH": Viseme.OO, "UW": Viseme.OO,
    "V": Viseme.FF, "W": Viseme.OO, "Y": Viseme.EE,
    "Z": Viseme.SS, "ZH": Viseme.SH,
}


# ── Emotional Visual States ──────────────────────────────────────────────────

class AvatarEmotion(Enum):
    IDLE        = "idle"
    SPEAKING    = "speaking"
    LISTENING   = "listening"
    CUTE_ANGER  = "cute_anger"
    FLIRTY      = "flirty"
    FOCUSED     = "focused"
    EMPATHIC    = "empathic"
    EXCITED     = "excited"


@dataclass
class EmotionVisuals:
    """Visual rendering parameters for each emotional state."""
    core_color: Tuple[int, int, int]          # RGB base color
    glow_color: Tuple[int, int, int]          # Outer glow RGB
    pulse_rate: float                          # Hz — how fast the core pulses
    particle_density: int                      # Ambient particle count
    expression_key: str                        # Animation expression name
    mouth_modifier: float = 1.0               # Lip-sync amplitude multiplier
    blush_intensity: float = 0.0              # 0.0–1.0 blush overlay
    eye_scale: float = 1.0                    # Eye size multiplier


EMOTION_VISUAL_MAP: Dict[AvatarEmotion, EmotionVisuals] = {
    AvatarEmotion.IDLE: EmotionVisuals(
        core_color=(0, 200, 255), glow_color=(0, 120, 200),
        pulse_rate=0.5, particle_density=20, expression_key="neutral",
    ),
    AvatarEmotion.SPEAKING: EmotionVisuals(
        core_color=(0, 240, 255), glow_color=(0, 180, 220),
        pulse_rate=2.0, particle_density=40, expression_key="talking",
        mouth_modifier=1.2,
    ),
    AvatarEmotion.LISTENING: EmotionVisuals(
        core_color=(80, 220, 255), glow_color=(40, 160, 200),
        pulse_rate=1.0, particle_density=30, expression_key="attentive",
        eye_scale=1.15,
    ),
    AvatarEmotion.CUTE_ANGER: EmotionVisuals(
        core_color=(255, 140, 50), glow_color=(255, 100, 30),
        pulse_rate=3.0, particle_density=60, expression_key="pout",
        mouth_modifier=0.5, eye_scale=1.3,
    ),
    AvatarEmotion.FLIRTY: EmotionVisuals(
        core_color=(200, 100, 255), glow_color=(160, 60, 220),
        pulse_rate=0.8, particle_density=50, expression_key="wink",
        blush_intensity=0.6, mouth_modifier=0.9,
    ),
    AvatarEmotion.FOCUSED: EmotionVisuals(
        core_color=(0, 255, 200), glow_color=(0, 200, 160),
        pulse_rate=1.5, particle_density=15, expression_key="determined",
        eye_scale=0.9,
    ),
    AvatarEmotion.EMPATHIC: EmotionVisuals(
        core_color=(255, 180, 100), glow_color=(220, 140, 60),
        pulse_rate=0.4, particle_density=35, expression_key="caring",
        eye_scale=1.2, blush_intensity=0.2,
    ),
    AvatarEmotion.EXCITED: EmotionVisuals(
        core_color=(255, 220, 0), glow_color=(255, 180, 0),
        pulse_rate=4.0, particle_density=80, expression_key="wide_smile",
        mouth_modifier=1.5, eye_scale=1.25,
    ),
}


# ── Particle System ──────────────────────────────────────────────────────────

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: float
    color: Tuple[int, int, int]


class HolographicParticleSystem:
    """GPU-inspired ambient floating particle emitter around the avatar core."""

    def __init__(self, max_particles: int = 100):
        self.max_particles = max_particles
        self.particles: List[Particle] = []
        self._time = 0.0

    def emit(self, cx: float, cy: float, count: int,
             color: Tuple[int, int, int]) -> None:
        """Emit `count` particles from center (cx, cy)."""
        import random
        for _ in range(min(count, self.max_particles - len(self.particles))):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.3, 2.0)
            life = random.uniform(1.0, 3.0)
            self.particles.append(Particle(
                x=cx, y=cy,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
                life=life, max_life=life,
                size=random.uniform(1.5, 4.0),
                color=color,
            ))

    def tick(self, dt: float) -> List[Dict[str, Any]]:
        """Advance simulation by `dt` seconds; return render data."""
        self._time += dt
        alive = []
        render_data = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.x += p.vx * dt * 30
            p.y += p.vy * dt * 30
            alpha = max(0.0, p.life / p.max_life)
            render_data.append({
                "x": p.x, "y": p.y,
                "size": p.size * alpha,
                "color": p.color,
                "alpha": alpha,
            })
            alive.append(p)
        self.particles = alive
        return render_data

    def clear(self) -> None:
        self.particles.clear()


# ── Cursor / Gaze Tracker ────────────────────────────────────────────────────

class CursorGazeTracker:
    """Tracks mouse cursor position for avatar eye tracking."""

    def __init__(self):
        self._cursor_pos: Tuple[int, int] = (0, 0)
        self._lock = threading.Lock()

    def update(self) -> Tuple[int, int]:
        """Get current cursor position using Win32 API."""
        try:
            import ctypes
            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            pt = POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            with self._lock:
                self._cursor_pos = (pt.x, pt.y)
        except Exception:
            pass
        return self._cursor_pos

    @property
    def position(self) -> Tuple[int, int]:
        with self._lock:
            return self._cursor_pos

    def compute_eye_offset(self, avatar_center: Tuple[float, float],
                           max_offset: float = 8.0) -> Tuple[float, float]:
        """Compute (dx, dy) offset for avatar eyes toward cursor."""
        cx, cy = avatar_center
        mx, my = self.position
        dx, dy = mx - cx, my - cy
        dist = math.sqrt(dx * dx + dy * dy) or 1.0
        scale = min(max_offset / dist, 1.0)
        return dx * scale, dy * scale


# ── Lip-Sync Analyzer ────────────────────────────────────────────────────────

class AudioToVisemeAnalyzer:
    """Maps streaming audio energy to real-time viseme frames."""

    def __init__(self):
        self._current_viseme: Viseme = Viseme.SILENCE
        self._energy: float = 0.0
        self._frame_count: int = 0

    def analyze_chunk(self, pcm_bytes: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """Analyze a raw PCM audio chunk and return viseme + energy."""
        self._frame_count += 1
        if not pcm_bytes:
            self._current_viseme = Viseme.SILENCE
            self._energy = 0.0
            return {"viseme": Viseme.SILENCE, "energy": 0.0, "frame": self._frame_count}

        # Compute RMS energy from 16-bit PCM
        import struct
        n_samples = len(pcm_bytes) // 2
        if n_samples == 0:
            self._energy = 0.0
            self._current_viseme = Viseme.SILENCE
            return {"viseme": Viseme.SILENCE, "energy": 0.0, "frame": self._frame_count}

        samples = struct.unpack(f"<{n_samples}h", pcm_bytes[:n_samples * 2])
        rms = math.sqrt(sum(s * s for s in samples) / n_samples)
        self._energy = min(rms / 3000.0, 1.0)  # Normalize to 0-1

        # Energy-based viseme estimation (simplified without full phoneme model)
        if self._energy < 0.05:
            self._current_viseme = Viseme.SILENCE
        elif self._energy < 0.2:
            self._current_viseme = Viseme.MM
        elif self._energy < 0.4:
            self._current_viseme = Viseme.EE
        elif self._energy < 0.6:
            self._current_viseme = Viseme.AA
        elif self._energy < 0.8:
            self._current_viseme = Viseme.OH
        else:
            self._current_viseme = Viseme.WIDE_OPEN

        return {
            "viseme": self._current_viseme,
            "energy": round(self._energy, 3),
            "frame": self._frame_count,
        }

    def map_phoneme(self, phoneme: str) -> Viseme:
        """Direct phoneme string to Viseme mapping."""
        return PHONEME_TO_VISEME.get(phoneme.upper(), Viseme.SILENCE)

    @property
    def current_viseme(self) -> Viseme:
        return self._current_viseme

    @property
    def energy(self) -> float:
        return self._energy


# ── Holographic Avatar Engine ─────────────────────────────────────────────────

class HolographicAvatarEngine:
    """
    Master controller for the desktop holographic avatar.
    
    Manages:
    - Avatar emotional state and visual rendering parameters
    - Real-time lip-sync from audio stream
    - Eye/cursor tracking
    - Particle effects
    - Position and visibility on desktop
    """

    def __init__(self, initial_emotion: AvatarEmotion = AvatarEmotion.IDLE):
        self._emotion = initial_emotion
        self._visuals = EMOTION_VISUAL_MAP[initial_emotion]
        self._particles = HolographicParticleSystem(max_particles=120)
        self._lip_sync = AudioToVisemeAnalyzer()
        self._cursor_tracker = CursorGazeTracker()
        self._visible = False
        self._position: Tuple[int, int] = (100, 100)  # Screen position
        self._size: int = 200  # Avatar diameter
        self._lock = threading.Lock()
        self._render_thread: Optional[threading.Thread] = None
        self._running = False
        self._fps = 30.0
        self._opacity = 0.85
        self._animation_time = 0.0
        logger.info("HolographicAvatarEngine initialized (emotion=%s)", initial_emotion.value)

    # ── Emotion Control ──────────────────────────────────────────────────

    def set_emotion(self, emotion: AvatarEmotion) -> Dict[str, Any]:
        """Transition avatar to a new emotional visual state."""
        with self._lock:
            prev = self._emotion
            self._emotion = emotion
            self._visuals = EMOTION_VISUAL_MAP.get(emotion, EMOTION_VISUAL_MAP[AvatarEmotion.IDLE])
            logger.info("Avatar emotion: %s → %s", prev.value, emotion.value)
            return {
                "previous": prev.value,
                "current": emotion.value,
                "core_color": self._visuals.core_color,
                "glow_color": self._visuals.glow_color,
                "expression": self._visuals.expression_key,
            }

    def get_emotion(self) -> AvatarEmotion:
        with self._lock:
            return self._emotion

    # ── Lip-Sync ─────────────────────────────────────────────────────────

    def feed_audio(self, pcm_chunk: bytes) -> Dict[str, Any]:
        """Feed raw PCM audio for lip-sync analysis."""
        result = self._lip_sync.analyze_chunk(pcm_chunk)
        # Scale by emotion modifier
        with self._lock:
            result["mouth_amplitude"] = result["energy"] * self._visuals.mouth_modifier
        return result

    def set_phoneme(self, phoneme: str) -> Viseme:
        """Direct phoneme input for precise lip-sync."""
        return self._lip_sync.map_phoneme(phoneme)

    # ── Eye Tracking ─────────────────────────────────────────────────────

    def update_cursor_tracking(self) -> Tuple[float, float]:
        """Update cursor position and compute eye offset."""
        self._cursor_tracker.update()
        center = (self._position[0] + self._size / 2,
                  self._position[1] + self._size / 2)
        return self._cursor_tracker.compute_eye_offset(center)

    # ── Render Frame ─────────────────────────────────────────────────────

    def render_frame(self, dt: float = 1 / 30) -> Dict[str, Any]:
        """Generate one frame of avatar render data."""
        self._animation_time += dt

        with self._lock:
            emotion = self._emotion
            visuals = self._visuals

        # Pulse calculation
        pulse = 0.5 + 0.5 * math.sin(self._animation_time * visuals.pulse_rate * 2 * math.pi)

        # Eye offset
        eye_dx, eye_dy = self.update_cursor_tracking()

        # Particles
        center = (self._position[0] + self._size / 2,
                  self._position[1] + self._size / 2)
        if len(self._particles.particles) < visuals.particle_density:
            self._particles.emit(center[0], center[1],
                                 visuals.particle_density // 5,
                                 visuals.glow_color)
        particle_data = self._particles.tick(dt)

        return {
            "emotion": emotion.value,
            "position": self._position,
            "size": self._size,
            "core_color": visuals.core_color,
            "glow_color": visuals.glow_color,
            "pulse_intensity": round(pulse, 3),
            "expression": visuals.expression_key,
            "eye_offset": (round(eye_dx, 1), round(eye_dy, 1)),
            "eye_scale": visuals.eye_scale,
            "blush_intensity": visuals.blush_intensity,
            "mouth_viseme": self._lip_sync.current_viseme.value,
            "mouth_energy": round(self._lip_sync.energy, 3),
            "particles": particle_data,
            "opacity": self._opacity,
            "timestamp": self._animation_time,
        }

    # ── Visibility & Position ────────────────────────────────────────────

    def show(self) -> bool:
        """Make the avatar visible on screen."""
        with self._lock:
            self._visible = True
        logger.info("Avatar shown at position %s", self._position)
        return True

    def hide(self) -> bool:
        """Hide the avatar from screen."""
        with self._lock:
            self._visible = False
        self._particles.clear()
        logger.info("Avatar hidden")
        return True

    def set_position(self, x: int, y: int) -> Tuple[int, int]:
        """Reposition the avatar on screen."""
        with self._lock:
            self._position = (x, y)
        return self._position

    def set_size(self, diameter: int) -> int:
        """Resize the avatar."""
        with self._lock:
            self._size = max(80, min(diameter, 600))
        return self._size

    def set_opacity(self, opacity: float) -> float:
        """Set avatar transparency (0.0 = invisible, 1.0 = fully opaque)."""
        with self._lock:
            self._opacity = max(0.0, min(opacity, 1.0))
        return self._opacity

    @property
    def visible(self) -> bool:
        with self._lock:
            return self._visible

    # ── Animation Loop ───────────────────────────────────────────────────

    def start_render_loop(self, callback: Optional[Callable] = None) -> bool:
        """Start background render loop at target FPS."""
        if self._running:
            return False
        self._running = True
        self._visible = True

        def _loop():
            dt = 1.0 / self._fps
            while self._running:
                frame = self.render_frame(dt)
                if callback:
                    try:
                        callback(frame)
                    except Exception as e:
                        logger.error("Render callback error: %s", e)
                time.sleep(dt)

        self._render_thread = threading.Thread(target=_loop, daemon=True, name="AvatarRender")
        self._render_thread.start()
        logger.info("Avatar render loop started at %.0f FPS", self._fps)
        return True

    def stop_render_loop(self) -> bool:
        """Stop the background render loop."""
        self._running = False
        if self._render_thread:
            self._render_thread.join(timeout=2.0)
            self._render_thread = None
        self._visible = False
        self._particles.clear()
        logger.info("Avatar render loop stopped")
        return True

    # ── Status ───────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Return full avatar engine status."""
        with self._lock:
            return {
                "visible": self._visible,
                "emotion": self._emotion.value,
                "position": self._position,
                "size": self._size,
                "opacity": self._opacity,
                "render_loop_active": self._running,
                "fps": self._fps,
                "lip_sync_viseme": self._lip_sync.current_viseme.value,
                "particle_count": len(self._particles.particles),
            }
