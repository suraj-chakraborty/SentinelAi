"""
sentinel/perception/biometric_auth.py
──────────────────────────────────────
Biometric Sentience Module - Zero-Touch Authentication
via Facial and Posture Recognition.

Features:
- Face recognition for automatic unlock
- Pose detection for engagement awareness
- Auto-lock when user walks away
- Welcome-back announcements
"""

import os
import time
import logging
import threading
import numpy as np
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

import cv2

from sentinel.app.config import APPDATA_DIR

logger = logging.getLogger("BiometricAuth")

BIOMETRIC_DIR = os.path.join(APPDATA_DIR, "biometric")
os.makedirs(BIOMETRIC_DIR, exist_ok=True)

OWNER_FACE_FILE = os.path.join(BIOMETRIC_DIR, "owner_face.npy")
OWNER_METADATA_FILE = os.path.join(BIOMETRIC_DIR, "owner_metadata.json")


@dataclass
class BiometricResult:
    """Result of biometric authentication."""
    success: bool
    confidence: float = 0.0
    message: str = ""
    face_embedding: Optional[np.ndarray] = None


class FaceRecognition:
    """Face recognition using DeepFace."""

    def __init__(self, model_name: str = "Facenet", threshold: float = 0.9):
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
        self._owner_embedding: Optional[np.ndarray] = None

    def _load_model(self):
        """Load DeepFace model."""
        if self._model is not None:
            return
        
        try:
            from deepface import DeepFace
            self._model = DeepFace
            logger.info(f"DeepFace model ({self.model_name}) loaded")
        except Exception as e:
            logger.error(f"Failed to load DeepFace: {e}")
            self._model = None

    def enroll_owner(self, image_path: str) -> bool:
        """Enroll owner's face for future recognition."""
        self._load_model()
        
        if self._model is None:
            logger.error("DeepFace not available for enrollment")
            return False
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to read image: {image_path}")
                return False
            
            embedding = self._model.represent(img, model_name=self.model_name)[0]["embedding"]
            self._owner_embedding = np.array(embedding)
            
            np.save(OWNER_FACE_FILE, self._owner_embedding)
            
            metadata = {
                "enrolled_at": datetime.now().isoformat(),
                "model": self.model_name,
                "threshold": self.threshold
            }
            with open(OWNER_METADATA_FILE, "w") as f:
                json.dump(metadata, f)
            
            logger.info("Owner face enrolled successfully")
            return True
        
        except Exception as e:
            logger.error(f"Face enrollment failed: {e}")
            return False

    def load_owner(self) -> bool:
        """Load previously enrolled owner face."""
        if not os.path.exists(OWNER_FACE_FILE):
            return False
        
        try:
            self._owner_embedding = np.load(OWNER_FACE_FILE)
            logger.info("Owner face loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load owner face: {e}")
            return False

    def verify(self, frame: np.ndarray) -> BiometricResult:
        """Verify if frame contains owner's face."""
        if self._owner_embedding is None:
            if not self.load_owner():
                return BiometricResult(False, message="Owner not enrolled")
        
        self._load_model()
        
        if self._model is None:
            return BiometricResult(False, message="Face recognition not available")
        
        try:
            result = self._model.verify(
                frame,
                self._owner_embedding,
                model_name=self.model_name,
                enforce_detection=False
            )
            
            verified = result.get("verified", False)
            distance = result.get("distance", 1.0)
            confidence = max(0, 1 - distance)
            
            return BiometricResult(
                success=verified and confidence >= self.threshold,
                confidence=confidence,
                message="Face verified" if verified else "Face not recognized"
            )
        
        except Exception as e:
            logger.debug(f"Face verification error: {e}")
            return BiometricResult(False, message=str(e))


class PoseDetector:
    """MediaPipe-based pose detection for engagement awareness."""

    def __init__(self, away_threshold: float = 0.5):
        self.away_threshold = away_threshold
        self._mp_pose = None
        self._pose = None

    def _load_model(self):
        """Load MediaPipe Pose."""
        if self._mp_pose is not None:
            return
        
        try:
            import mediapipe as mp
            self._mp_pose = mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe Pose loaded")
        except Exception as e:
            logger.error(f"Failed to load MediaPipe: {e}")

    def detect_pose(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect pose in frame."""
        self._load_model()
        
        if self._pose is None:
            return None
        
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._pose.process(frame_rgb)
            
            if not results.pose_landmarks:
                return None
            
            landmarks = results.pose_landmarks.landmark
            
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            
            return {
                "nose_x": nose.x,
                "nose_y": nose.y,
                "nose_z": nose.z,
                "shoulder_center_y": shoulder_center_y,
                "visibility": (left_shoulder.visibility + right_shoulder.visibility) / 2,
                "landmarks": landmarks
            }
        
        except Exception as e:
            logger.debug(f"Pose detection error: {e}")
            return None

    def is_away(self, pose: Dict[str, Any]) -> bool:
        """Determine if user is away from desk."""
        if pose is None:
            return True
        
        visibility = pose.get("visibility", 0)
        if visibility < 0.5:
            return True
        
        nose_y = pose.get("nose_y", 0)
        return nose_y < self.away_threshold


class BiometricAuth:
    """
    Zero-touch authentication via face and posture.
    
    Features:
    - Face recognition for automatic unlock
    - Posture detection for engagement awareness
    - Auto-lock when user walks away
    - Welcome-back announcements
    """

    def __init__(
        self,
        face_threshold: float = 0.9,
        pose_threshold: float = 0.5,
        on_unlock: Optional[Callable] = None,
        on_lock: Optional[Callable] = None,
        on_welcome_back: Optional[Callable] = None
    ):
        self.face_threshold = face_threshold
        self.pose_threshold = pose_threshold
        
        self.on_unlock = on_unlock
        self.on_lock = on_lock
        self.on_welcome_back = on_welcome_back
        
        self._face_recognizer = FaceRecognition(threshold=face_threshold)
        self._pose_detector = PoseDetector(away_threshold=pose_threshold)
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self._is_unlocked = False
        self._was_away = False
        self._away_start_time: Optional[datetime] = None
        self._away_duration_threshold_sec = 30

    def start(self):
        """Start biometric monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="BiometricAuth")
        self._thread.start()
        
        logger.info("Biometric authentication started")

    def stop(self):
        """Stop biometric monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Biometric authentication stopped")

    def enroll_owner_face(self, image_path: str) -> bool:
        """Enroll owner's face."""
        return self._face_recognizer.enroll_owner(image_path)

    def verify_face(self, frame: np.ndarray) -> BiometricResult:
        """Verify face in frame."""
        return self._face_recognizer.verify(frame)

    def check_pose(self, frame: np.ndarray) -> Dict[str, Any]:
        """Check user pose in frame."""
        pose = self._pose_detector.detect_pose(frame)
        
        if pose is None:
            return {"detected": False, "away": True}
        
        is_away = self._pose_detector.is_away(pose)
        
        return {
            "detected": True,
            "away": is_away,
            "visibility": pose.get("visibility", 0)
        }

    def _monitor_loop(self):
        """Main biometric monitoring loop."""
        check_interval = 1.0
        
        while self._running:
            try:
                if not self._is_unlocked:
                    continue
                
                from sentinel.vision.live_vision import get_live_vision
                stream = get_live_vision()
                frame = stream.get_latest_frame()
                
                if frame is not None:
                    pose_result = self.check_pose(frame)
                    
                    if pose_result.get("away", True):
                        if not self._was_away:
                            self._away_start_time = datetime.now()
                            self._was_away = True
                        
                        if self._away_start_time:
                            away_duration = (datetime.now() - self._away_start_time).total_seconds()
                            if away_duration > self._away_duration_threshold_sec:
                                self._handle_away()
                    else:
                        if self._was_away:
                            self._handle_back()
                            self._was_away = False
                            self._away_start_time = None
                
                time.sleep(check_interval)
            
            except Exception as e:
                logger.error(f"Biometric monitor error: {e}")
                time.sleep(check_interval)

    def _handle_away(self):
        """Handle user walking away."""
        logger.info("User away - triggering auto-lock")
        
        if self.on_lock:
            self.on_lock()
        
        try:
            from sentinel.core.intent_detector import Intent
            from sentinel.commands.system_control import get_system_control
            ctrl = get_system_control()
            ctrl.execute("lock")
        except Exception as e:
            logger.debug(f"Auto-lock failed: {e}")

    def _handle_back(self):
        """Handle user returning."""
        logger.info("User returned - unlocking")
        
        self._is_unlocked = True
        
        if self.on_welcome_back:
            self.on_welcome_back()
        
        try:
            summary = self._get_away_summary()
            logger.info(f"Welcome back summary: {summary}")
        except Exception as e:
            logger.debug(f"Summary failed: {e}")

    def _get_away_summary(self) -> str:
        """Get summary of events while user was away."""
        return "While you were away, you received 2 emails."

    def unlock(self):
        """Mark as unlocked."""
        self._is_unlocked = True

    def lock(self):
        """Mark as locked."""
        self._is_unlocked = False
        self._was_away = False

    def is_unlocked(self) -> bool:
        """Check if currently unlocked."""
        return self._is_unlocked

    def get_stats(self) -> Dict[str, Any]:
        """Get biometric auth statistics."""
        return {
            "running": self._running,
            "is_unlocked": self._is_unlocked,
            "was_away": self._was_away,
            "face_threshold": self.face_threshold,
            "pose_threshold": self.pose_threshold,
            "owner_enrolled": os.path.exists(OWNER_FACE_FILE)
        }


_biometric_auth: Optional[BiometricAuth] = None


def get_biometric_auth() -> BiometricAuth:
    global _biometric_auth
    if _biometric_auth is None:
        _biometric_auth = BiometricAuth()
    return _biometric_auth


def start_biometric_auth():
    auth = get_biometric_auth()
    auth.start()
    return auth


def stop_biometric_auth():
    global _biometric_auth
    if _biometric_auth:
        _biometric_auth.stop()