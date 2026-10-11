"""
sentinel/perception/biometric_vitals.py
────────────────────────────────────────
Biometric Vitals Dashboard & Wellness Monitor.

Provides continuous health/ergonomic monitoring:
  1. Heart Rate Estimation via webcam rPPG (remote photoplethysmography).
  2. Posture & Head Angle monitoring (slouch detection).
  3. Eye Strain / Blink Rate tracking.
  4. Stress Index estimation from HRV (heart rate variability).
  5. Hydration & Break reminders based on activity duration.
  6. Daily wellness score computation.
  7. Historical vitals logging for trend analysis.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("BiometricVitals")

_APPDATA = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi")


class VitalStatus(Enum):
    HEALTHY     = "healthy"
    CAUTION     = "caution"
    WARNING     = "warning"
    CRITICAL    = "critical"
    UNKNOWN     = "unknown"


class PostureState(Enum):
    UPRIGHT     = "upright"
    SLIGHT_LEAN = "slight_lean"
    SLOUCHING   = "slouching"
    UNKNOWN     = "unknown"


@dataclass
class VitalSnapshot:
    """A single point-in-time biometric reading."""
    timestamp: float
    heart_rate_bpm: Optional[float] = None
    blink_rate_per_min: Optional[float] = None
    posture: PostureState = PostureState.UNKNOWN
    head_angle_deg: Optional[float] = None
    stress_index: Optional[float] = None       # 0.0 = calm, 1.0 = very stressed
    screen_time_minutes: float = 0.0
    last_break_minutes_ago: float = 0.0
    ambient_light_lux: Optional[float] = None


@dataclass
class WellnessScore:
    """Computed daily wellness score and breakdown."""
    overall: float              # 0–100
    heart_health: float         # 0–100
    eye_health: float           # 0–100
    posture_score: float        # 0–100
    stress_score: float         # 0–100 (100 = very calm)
    break_compliance: float     # 0–100
    timestamp: str = ""


class HeartRateEstimator:
    """
    Remote Photoplethysmography (rPPG) heart rate estimator.
    Analyzes subtle skin color changes in webcam video to estimate pulse.
    """

    def __init__(self, buffer_seconds: int = 10, fps: float = 30.0):
        self._buffer_size = int(buffer_seconds * fps)
        self._green_channel_buffer: deque = deque(maxlen=self._buffer_size)
        self._fps = fps
        self._last_hr: Optional[float] = None
        self._hrv_buffer: deque = deque(maxlen=30)

    def add_frame_roi(self, green_mean: float) -> None:
        """Add a mean green channel value from the forehead ROI."""
        self._green_channel_buffer.append(green_mean)

    def estimate_heart_rate(self) -> Optional[float]:
        """Estimate heart rate from green channel oscillations using peak detection."""
        if len(self._green_channel_buffer) < self._buffer_size * 0.7:
            return self._last_hr

        signal = list(self._green_channel_buffer)
        n = len(signal)

        # Simple bandpass: remove DC component
        mean_val = sum(signal) / n
        detrended = [s - mean_val for s in signal]

        # Count zero crossings (rough frequency estimation)
        crossings = 0
        for i in range(1, n):
            if detrended[i - 1] * detrended[i] < 0:
                crossings += 1

        # Crossings to BPM: freq = crossings / (2 * duration_seconds)
        duration_s = n / self._fps
        freq_hz = crossings / (2.0 * duration_s)
        bpm = freq_hz * 60.0

        # Physiological bounds
        if 40 <= bpm <= 180:
            self._last_hr = round(bpm, 1)
            self._hrv_buffer.append(bpm)
        elif self._last_hr:
            pass  # Keep last valid reading

        return self._last_hr

    def get_stress_index(self) -> Optional[float]:
        """Estimate stress from heart rate variability (RMSSD proxy)."""
        if len(self._hrv_buffer) < 5:
            return None
        diffs = [abs(self._hrv_buffer[i] - self._hrv_buffer[i - 1])
                 for i in range(1, len(self._hrv_buffer))]
        rmssd = math.sqrt(sum(d * d for d in diffs) / len(diffs))
        # Lower RMSSD = higher stress (inverted and normalized)
        stress = max(0.0, min(1.0, 1.0 - (rmssd / 20.0)))
        return round(stress, 2)


class BlinkTracker:
    """Tracks eye blink rate for eye strain assessment."""

    def __init__(self):
        self._blink_timestamps: deque = deque(maxlen=100)
        self._eye_aspect_ratios: deque = deque(maxlen=10)
        self._blink_threshold = 0.25
        self._was_closed = False

    def update_eye_aspect_ratio(self, ear: float) -> bool:
        """Update with current eye aspect ratio. Returns True if blink detected."""
        self._eye_aspect_ratios.append(ear)
        blink_detected = False

        if ear < self._blink_threshold:
            self._was_closed = True
        elif self._was_closed:
            self._was_closed = False
            blink_detected = True
            self._blink_timestamps.append(time.time())

        return blink_detected

    def get_blink_rate(self, window_seconds: float = 60.0) -> float:
        """Calculate blinks per minute over recent window."""
        now = time.time()
        recent = [t for t in self._blink_timestamps if now - t <= window_seconds]
        if len(recent) < 2:
            return 0.0
        duration = now - recent[0]
        if duration <= 0:
            return 0.0
        return round(len(recent) / (duration / 60.0), 1)


class PostureAnalyzer:
    """Analyzes head/neck posture for ergonomic monitoring."""

    def __init__(self):
        self._current_posture = PostureState.UNKNOWN
        self._head_angle = 0.0
        self._slouch_threshold = 15.0    # degrees
        self._severe_threshold = 30.0

    def analyze_head_pose(self, pitch: float, yaw: float, roll: float) -> PostureState:
        """Analyze head angles and return posture state."""
        self._head_angle = abs(pitch)

        if abs(pitch) < self._slouch_threshold:
            self._current_posture = PostureState.UPRIGHT
        elif abs(pitch) < self._severe_threshold:
            self._current_posture = PostureState.SLIGHT_LEAN
        else:
            self._current_posture = PostureState.SLOUCHING

        return self._current_posture

    def analyze_shoulder_ratio(self, left_y: float, right_y: float) -> PostureState:
        """Analyze shoulder alignment from landmark positions."""
        diff = abs(left_y - right_y)
        if diff < 0.03:
            return PostureState.UPRIGHT
        elif diff < 0.06:
            return PostureState.SLIGHT_LEAN
        return PostureState.SLOUCHING

    @property
    def current_posture(self) -> PostureState:
        return self._current_posture

    @property
    def head_angle(self) -> float:
        return self._head_angle


class BiometricVitalsEngine:
    """
    Master biometric vitals controller.

    Coordinates heart rate estimation, blink tracking, posture analysis,
    and generates wellness reports with break reminders.
    """

    def __init__(self, break_interval_min: float = 45.0,
                 alert_callback: Optional[Callable[[str], None]] = None):
        self._hr_estimator = HeartRateEstimator()
        self._blink_tracker = BlinkTracker()
        self._posture_analyzer = PostureAnalyzer()
        self._alert_callback = alert_callback
        self._break_interval = break_interval_min
        self._session_start = time.time()
        self._last_break_time = time.time()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._vitals_history: List[VitalSnapshot] = []
        self._max_history = 500
        self._hydration_reminders = 0
        self._posture_alerts = 0
        self._log_dir = os.path.join(_APPDATA, "vitals_logs")
        os.makedirs(self._log_dir, exist_ok=True)
        logger.info("BiometricVitalsEngine initialized (break_interval=%.0fmin)", break_interval_min)

    def start(self) -> bool:
        """Start biometric monitoring."""
        if self._running:
            return False
        self._running = True
        self._session_start = time.time()
        self._last_break_time = time.time()

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True,
                                         name="BiometricVitals")
        self._thread.start()
        logger.info("Biometric monitoring started")
        return True

    def stop(self) -> bool:
        """Stop biometric monitoring and save logs."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._save_session_log()
        logger.info("Biometric monitoring stopped")
        return True

    def _monitor_loop(self):
        """Background monitoring loop — checks break timers and posture."""
        while self._running:
            now = time.time()
            minutes_since_break = (now - self._last_break_time) / 60.0
            minutes_active = (now - self._session_start) / 60.0

            # Break reminder
            if minutes_since_break >= self._break_interval:
                self._send_alert(
                    f"🧘 You've been active for {int(minutes_since_break)} minutes. "
                    f"Time for a break! Stretch, hydrate, and rest your eyes."
                )
                self._hydration_reminders += 1

            # Record snapshot
            snapshot = VitalSnapshot(
                timestamp=now,
                heart_rate_bpm=self._hr_estimator._last_hr,
                blink_rate_per_min=self._blink_tracker.get_blink_rate(),
                posture=self._posture_analyzer.current_posture,
                head_angle_deg=self._posture_analyzer.head_angle,
                stress_index=self._hr_estimator.get_stress_index(),
                screen_time_minutes=minutes_active,
                last_break_minutes_ago=minutes_since_break,
            )
            self._vitals_history.append(snapshot)
            if len(self._vitals_history) > self._max_history:
                self._vitals_history.pop(0)

            # Posture alert
            if snapshot.posture == PostureState.SLOUCHING:
                self._posture_alerts += 1
                if self._posture_alerts % 3 == 1:  # Don't spam
                    self._send_alert(
                        "🪑 Posture check! You're slouching. Sit up straight and align your screen."
                    )

            time.sleep(30)  # Check every 30 seconds

    def _send_alert(self, message: str):
        if self._alert_callback:
            try:
                self._alert_callback(message)
            except Exception:
                pass
        logger.info("Vitals alert: %s", message[:80])

    def record_break(self) -> str:
        """User explicitly takes a break."""
        self._last_break_time = time.time()
        return "Break recorded! Timer reset. Remember to stretch and hydrate. 💧"

    def feed_camera_frame(self, green_mean: float, ear: Optional[float] = None,
                           head_pose: Optional[Tuple[float, float, float]] = None) -> Dict[str, Any]:
        """Feed camera data for real-time vitals estimation."""
        result = {}

        # Heart rate
        self._hr_estimator.add_frame_roi(green_mean)
        hr = self._hr_estimator.estimate_heart_rate()
        result["heart_rate_bpm"] = hr

        # Blink
        if ear is not None:
            blink = self._blink_tracker.update_eye_aspect_ratio(ear)
            result["blink_detected"] = blink
            result["blink_rate"] = self._blink_tracker.get_blink_rate()

        # Posture
        if head_pose:
            posture = self._posture_analyzer.analyze_head_pose(*head_pose)
            result["posture"] = posture.value
            result["head_angle"] = self._posture_analyzer.head_angle

        # Stress
        stress = self._hr_estimator.get_stress_index()
        result["stress_index"] = stress

        return result

    def compute_wellness_score(self) -> WellnessScore:
        """Compute current wellness score breakdown."""
        hr = self._hr_estimator._last_hr
        blink_rate = self._blink_tracker.get_blink_rate()
        stress = self._hr_estimator.get_stress_index()
        posture = self._posture_analyzer.current_posture
        mins_since_break = (time.time() - self._last_break_time) / 60.0

        # Heart health: optimal resting HR is 60-80
        if hr and 55 <= hr <= 90:
            heart_score = 90.0
        elif hr and 50 <= hr <= 100:
            heart_score = 70.0
        elif hr:
            heart_score = 50.0
        else:
            heart_score = 50.0  # Unknown

        # Eye health: normal blink rate is 15-20/min
        if blink_rate >= 15:
            eye_score = 90.0
        elif blink_rate >= 10:
            eye_score = 70.0
        elif blink_rate >= 5:
            eye_score = 50.0
        else:
            eye_score = 40.0

        # Posture
        posture_map = {
            PostureState.UPRIGHT: 95.0,
            PostureState.SLIGHT_LEAN: 65.0,
            PostureState.SLOUCHING: 30.0,
            PostureState.UNKNOWN: 50.0,
        }
        posture_score = posture_map.get(posture, 50.0)

        # Stress: 0=calm=high score, 1=stressed=low score
        stress_score = (1.0 - (stress or 0.5)) * 100.0

        # Break compliance
        if mins_since_break <= self._break_interval:
            break_score = 95.0
        elif mins_since_break <= self._break_interval * 1.5:
            break_score = 60.0
        else:
            break_score = 30.0

        overall = (heart_score * 0.25 + eye_score * 0.20 + posture_score * 0.20 +
                   stress_score * 0.20 + break_score * 0.15)

        return WellnessScore(
            overall=round(overall, 1),
            heart_health=round(heart_score, 1),
            eye_health=round(eye_score, 1),
            posture_score=round(posture_score, 1),
            stress_score=round(stress_score, 1),
            break_compliance=round(break_score, 1),
            timestamp=datetime.datetime.now().isoformat(),
        )

    def get_current_vitals(self) -> Dict[str, Any]:
        """Get latest vital signs snapshot."""
        return {
            "heart_rate_bpm": self._hr_estimator._last_hr,
            "blink_rate_per_min": self._blink_tracker.get_blink_rate(),
            "posture": self._posture_analyzer.current_posture.value,
            "head_angle_deg": self._posture_analyzer.head_angle,
            "stress_index": self._hr_estimator.get_stress_index(),
            "screen_time_min": round((time.time() - self._session_start) / 60.0, 1),
            "last_break_min_ago": round((time.time() - self._last_break_time) / 60.0, 1),
            "hydration_reminders": self._hydration_reminders,
            "posture_alerts": self._posture_alerts,
            "monitoring": self._running,
        }

    def _save_session_log(self):
        """Persist vitals session to disk."""
        try:
            fname = f"vitals_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            path = os.path.join(self._log_dir, fname)
            data = [{
                "timestamp": v.timestamp,
                "heart_rate": v.heart_rate_bpm,
                "blink_rate": v.blink_rate_per_min,
                "posture": v.posture.value,
                "stress": v.stress_index,
                "screen_time": v.screen_time_minutes,
            } for v in self._vitals_history[-100:]]
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Vitals session saved: %s", path)
        except Exception as e:
            logger.error("Failed to save vitals: %s", e)

    def get_status(self) -> Dict[str, Any]:
        return {
            "monitoring": self._running,
            "history_size": len(self._vitals_history),
            "session_minutes": round((time.time() - self._session_start) / 60.0, 1),
            "hydration_reminders": self._hydration_reminders,
            "posture_alerts": self._posture_alerts,
        }
