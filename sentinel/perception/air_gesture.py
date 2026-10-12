"""
sentinel/perception/air_gesture.py
───────────────────────────────────
Air-Gesture Recognition & Spatial Commands Engine.

Uses MediaPipe Hands (or falls back to mock) for real-time hand tracking:
  1. Swipe Left/Right/Up/Down — window snapping, tab switching
  2. Pinch-to-Zoom — UI scaling, map zoom
  3. Palm Hold (5s) — activate Sentinel listening
  4. Two-Finger Tap — pause/play media
  5. Fist Close — dismiss notification / stop TTS
  6. Circular Gesture — volume control
  7. Victory/Peace Sign — take screenshot
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("AirGesture")


class GestureType(Enum):
    NONE = "none"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    PINCH_IN = "pinch_in"
    PINCH_OUT = "pinch_out"
    PALM_HOLD = "palm_hold"
    FIST_CLOSE = "fist_close"
    TWO_FINGER_TAP = "two_finger_tap"
    CIRCLE_CW = "circle_clockwise"
    CIRCLE_CCW = "circle_counterclockwise"
    PEACE_SIGN = "peace_sign"
    THUMBS_UP = "thumbs_up"
    POINT = "point"


# Actions mapped to each gesture
GESTURE_ACTION_MAP: dict[GestureType, str] = {
    GestureType.SWIPE_LEFT: "snap_window_left",
    GestureType.SWIPE_RIGHT: "snap_window_right",
    GestureType.SWIPE_UP: "maximize_window",
    GestureType.SWIPE_DOWN: "minimize_window",
    GestureType.PINCH_IN: "zoom_out",
    GestureType.PINCH_OUT: "zoom_in",
    GestureType.PALM_HOLD: "activate_listening",
    GestureType.FIST_CLOSE: "dismiss_notification",
    GestureType.TWO_FINGER_TAP: "toggle_media",
    GestureType.CIRCLE_CW: "volume_up",
    GestureType.CIRCLE_CCW: "volume_down",
    GestureType.PEACE_SIGN: "take_screenshot",
    GestureType.THUMBS_UP: "confirm_action",
    GestureType.POINT: "click_at_position",
}


@dataclass
class HandLandmark:
    """A single tracked hand landmark (21 landmarks per hand)."""

    id: int
    x: float  # Normalized 0–1
    y: float
    z: float


@dataclass
class HandFrame:
    """One frame of hand tracking data."""

    landmarks: list[HandLandmark]
    handedness: str  # "Left" or "Right"
    confidence: float
    timestamp: float


@dataclass
class GestureEvent:
    """A detected gesture event."""

    gesture: GestureType
    action: str
    confidence: float
    hand: str
    position: tuple[float, float]
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


class GestureRecognizer:
    """
    Recognizes hand gestures from landmark streams.
    Uses velocity tracking, finger state analysis, and trajectory patterns.
    """

    def __init__(self):
        self._history: list[HandFrame] = []
        self._max_history = 30  # ~1 second at 30fps
        self._last_gesture_time = 0.0
        self._cooldown = 0.5  # seconds between repeated gestures
        self._palm_hold_start: float | None = None
        self._palm_hold_threshold = 2.0  # seconds

    def add_frame(self, frame: HandFrame) -> GestureEvent | None:
        """Process a hand frame and return gesture if detected."""
        self._history.append(frame)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        now = frame.timestamp
        if now - self._last_gesture_time < self._cooldown:
            return None

        # Check gestures by priority
        gesture = self._check_fist(frame)
        if not gesture:
            gesture = self._check_peace_sign(frame)
        if not gesture:
            gesture = self._check_thumbs_up(frame)
        if not gesture:
            gesture = self._check_palm_hold(frame, now)
        if not gesture:
            gesture = self._check_pinch(frame)
        if not gesture:
            gesture = self._check_swipe(now)

        if gesture:
            self._last_gesture_time = now
            return gesture
        return None

    def _finger_extended(self, landmarks: list[HandLandmark], tip_id: int, pip_id: int) -> bool:
        """Check if a finger is extended (tip above PIP joint)."""
        if tip_id >= len(landmarks) or pip_id >= len(landmarks):
            return False
        return landmarks[tip_id].y < landmarks[pip_id].y

    def _check_fist(self, frame: HandFrame) -> GestureEvent | None:
        """All fingers curled → fist close."""
        lm = frame.landmarks
        if len(lm) < 21:
            return None
        # Check all 4 finger tips (8,12,16,20) are below their PIP joints (6,10,14,18)
        fingers_curled = all(
            not self._finger_extended(lm, tip, pip) for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]
        )
        if fingers_curled:
            wrist = lm[0]
            return GestureEvent(
                gesture=GestureType.FIST_CLOSE,
                action=GESTURE_ACTION_MAP[GestureType.FIST_CLOSE],
                confidence=0.85,
                hand=frame.handedness,
                position=(wrist.x, wrist.y),
                timestamp=frame.timestamp,
            )
        return None

    def _check_peace_sign(self, frame: HandFrame) -> GestureEvent | None:
        """Index + Middle extended, Ring + Pinky curled → Victory/Peace."""
        lm = frame.landmarks
        if len(lm) < 21:
            return None
        index_up = self._finger_extended(lm, 8, 6)
        middle_up = self._finger_extended(lm, 12, 10)
        ring_down = not self._finger_extended(lm, 16, 14)
        pinky_down = not self._finger_extended(lm, 20, 18)
        if index_up and middle_up and ring_down and pinky_down:
            return GestureEvent(
                gesture=GestureType.PEACE_SIGN,
                action=GESTURE_ACTION_MAP[GestureType.PEACE_SIGN],
                confidence=0.88,
                hand=frame.handedness,
                position=(lm[0].x, lm[0].y),
                timestamp=frame.timestamp,
            )
        return None

    def _check_thumbs_up(self, frame: HandFrame) -> GestureEvent | None:
        """Thumb extended upward, all others curled → Thumbs Up."""
        lm = frame.landmarks
        if len(lm) < 21:
            return None
        thumb_up = lm[4].y < lm[3].y  # Thumb tip above thumb IP
        others_down = all(
            not self._finger_extended(lm, tip, pip) for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]
        )
        if thumb_up and others_down:
            return GestureEvent(
                gesture=GestureType.THUMBS_UP,
                action=GESTURE_ACTION_MAP[GestureType.THUMBS_UP],
                confidence=0.82,
                hand=frame.handedness,
                position=(lm[4].x, lm[4].y),
                timestamp=frame.timestamp,
            )
        return None

    def _check_palm_hold(self, frame: HandFrame, now: float) -> GestureEvent | None:
        """All fingers extended and held still → palm hold activation."""
        lm = frame.landmarks
        if len(lm) < 21:
            self._palm_hold_start = None
            return None
        all_extended = all(self._finger_extended(lm, tip, pip) for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)])
        if all_extended:
            if self._palm_hold_start is None:
                self._palm_hold_start = now
            elif now - self._palm_hold_start >= self._palm_hold_threshold:
                self._palm_hold_start = None
                return GestureEvent(
                    gesture=GestureType.PALM_HOLD,
                    action=GESTURE_ACTION_MAP[GestureType.PALM_HOLD],
                    confidence=0.90,
                    hand=frame.handedness,
                    position=(lm[0].x, lm[0].y),
                    timestamp=now,
                    metadata={"hold_duration": self._palm_hold_threshold},
                )
        else:
            self._palm_hold_start = None
        return None

    def _check_pinch(self, frame: HandFrame) -> GestureEvent | None:
        """Thumb tip close to index tip → pinch gesture."""
        lm = frame.landmarks
        if len(lm) < 21:
            return None
        thumb_tip = lm[4]
        index_tip = lm[8]
        dist = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
        if dist < 0.04:
            return GestureEvent(
                gesture=GestureType.PINCH_IN,
                action=GESTURE_ACTION_MAP[GestureType.PINCH_IN],
                confidence=0.80,
                hand=frame.handedness,
                position=((thumb_tip.x + index_tip.x) / 2, (thumb_tip.y + index_tip.y) / 2),
                timestamp=frame.timestamp,
                metadata={"distance": round(dist, 4)},
            )
        return None

    def _check_swipe(self, now: float) -> GestureEvent | None:
        """Check recent wrist trajectory for swipe gestures."""
        if len(self._history) < 8:
            return None
        recent = self._history[-8:]
        start_wrist = recent[0].landmarks[0] if recent[0].landmarks else None
        end_wrist = recent[-1].landmarks[0] if recent[-1].landmarks else None
        if not start_wrist or not end_wrist:
            return None

        dx = end_wrist.x - start_wrist.x
        dy = end_wrist.y - start_wrist.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.15:
            return None

        hand = recent[-1].handedness
        pos = (end_wrist.x, end_wrist.y)

        if abs(dx) > abs(dy):
            gesture_type = GestureType.SWIPE_RIGHT if dx > 0 else GestureType.SWIPE_LEFT
        else:
            gesture_type = GestureType.SWIPE_DOWN if dy > 0 else GestureType.SWIPE_UP

        return GestureEvent(
            gesture=gesture_type,
            action=GESTURE_ACTION_MAP[gesture_type],
            confidence=min(dist * 3, 0.95),
            hand=hand,
            position=pos,
            timestamp=now,
            metadata={"displacement": round(dist, 3), "dx": round(dx, 3), "dy": round(dy, 3)},
        )


class AirGestureController:
    """
    High-level controller that manages webcam capture, hand tracking,
    gesture recognition, and action dispatch.
    """

    def __init__(self, action_callback: Callable[[GestureEvent], None] | None = None):
        self._recognizer = GestureRecognizer()
        self._action_callback = action_callback
        self._running = False
        self._thread: threading.Thread | None = None
        self._gesture_log: list[GestureEvent] = []
        self._max_log = 50
        self._camera_index = 0
        self._enabled_gestures: set = set(GestureType)
        logger.info("AirGestureController initialized")

    def start(self) -> bool:
        """Start gesture recognition from webcam feed."""
        if self._running:
            return False
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="AirGestureCapture")
        self._thread.start()
        logger.info("Air gesture recognition started")
        return True

    def stop(self) -> bool:
        """Stop gesture recognition."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("Air gesture recognition stopped")
        return True

    def _capture_loop(self):
        """Main capture loop — uses MediaPipe Hands when available."""
        mp_hands = None
        cap = None
        try:
            import cv2
            import mediapipe as mp

            mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            cap = cv2.VideoCapture(self._camera_index)
        except ImportError:
            logger.warning("OpenCV/MediaPipe not available — running in simulation mode")

        try:
            while self._running:
                if cap and mp_hands:
                    ret, frame_img = cap.read()
                    if not ret:
                        time.sleep(0.03)
                        continue
                    import cv2

                    rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                    result = mp_hands.process(rgb)
                    if result.multi_hand_landmarks:
                        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                            landmarks = [
                                HandLandmark(
                                    id=i,
                                    x=lm.x,
                                    y=lm.y,
                                    z=lm.z,
                                )
                                for i, lm in enumerate(hand_landmarks.landmark)
                            ]
                            hand_frame = HandFrame(
                                landmarks=landmarks,
                                handedness=hand_info.classification[0].label,
                                confidence=hand_info.classification[0].score,
                                timestamp=time.time(),
                            )
                            gesture = self._recognizer.add_frame(hand_frame)
                            if gesture and gesture.gesture in self._enabled_gestures:
                                self._log_gesture(gesture)
                                if self._action_callback:
                                    self._action_callback(gesture)
                else:
                    # Simulation mode — sleep and do nothing
                    time.sleep(0.5)
        finally:
            if cap:
                cap.release()
            if mp_hands:
                mp_hands.close()

    def _log_gesture(self, event: GestureEvent):
        self._gesture_log.append(event)
        if len(self._gesture_log) > self._max_log:
            self._gesture_log.pop(0)
        logger.info("Gesture detected: %s (action=%s, conf=%.2f)", event.gesture.value, event.action, event.confidence)

    def process_frame_landmarks(self, landmarks_data: list[dict], handedness: str = "Right") -> GestureEvent | None:
        """Manually feed landmark data for testing or external tracking."""
        landmarks = [HandLandmark(id=lm["id"], x=lm["x"], y=lm["y"], z=lm.get("z", 0.0)) for lm in landmarks_data]
        frame = HandFrame(
            landmarks=landmarks,
            handedness=handedness,
            confidence=1.0,
            timestamp=time.time(),
        )
        gesture = self._recognizer.add_frame(frame)
        if gesture:
            self._log_gesture(gesture)
            if self._action_callback:
                self._action_callback(gesture)
        return gesture

    def get_recent_gestures(self, count: int = 10) -> list[dict[str, Any]]:
        """Return recent gesture events."""
        return [
            {
                "gesture": g.gesture.value,
                "action": g.action,
                "confidence": round(g.confidence, 2),
                "hand": g.hand,
                "timestamp": g.timestamp,
            }
            for g in self._gesture_log[-count:]
        ]

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "total_gestures_detected": len(self._gesture_log),
            "recent_gesture": self._gesture_log[-1].gesture.value if self._gesture_log else "none",
            "camera_index": self._camera_index,
        }

    @property
    def is_running(self) -> bool:
        return self._running
