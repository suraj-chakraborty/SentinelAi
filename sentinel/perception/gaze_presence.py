"""
sentinel/perception/gaze_presence.py
────────────────────────────────────
Webcam Desk Presence & Gaze-Wake Detection.
Detects user presence at the desk, locks workstation on prolonged absence,
and flags when the user is directly looking at Sentinel.
"""

import threading
import time
import sys
import logging
from typing import Optional, Callable

logger = logging.getLogger("GazePresence")

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class GazePresenceDetector:
    """
    Background webcam monitor tracking face presence and gaze alignment.
    """

    def __init__(
        self,
        camera_id: int = 0,
        absence_timeout: float = 120.0,
        auto_lock_enabled: bool = False,
        on_absence: Optional[Callable[[], None]] = None,
        on_presence_restored: Optional[Callable[[], None]] = None
    ):
        self.camera_id = camera_id
        self.absence_timeout = absence_timeout
        self.auto_lock_enabled = auto_lock_enabled
        self.on_absence = on_absence
        self.on_presence_restored = on_presence_restored

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._user_present = True
        self._gaze_engaged = False
        self._last_seen = time.time()
        self._locked = False

    @property
    def is_present(self) -> bool:
        return self._user_present

    @property
    def is_gaze_engaged(self) -> bool:
        return self._gaze_engaged

    def start(self):
        """Start presence monitoring loop."""
        if self._running or not _CV2_AVAILABLE:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="GazePresenceLoop")
        self._thread.start()
        logger.info("GazePresenceDetector started.")

    def stop(self):
        """Stop presence loop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("GazePresenceDetector stopped.")

    def lock_workstation(self):
        """Invoke Windows LockWorkStation API."""
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.user32.LockWorkStation()
                self._locked = True
                logger.info("Workstation locked by GazePresenceDetector.")
            except Exception as e:
                logger.debug(f"LockWorkStation error: {e}")

    def _monitor_loop(self):
        cap = None
        try:
            # Load standard OpenCV Haar cascade for lightweight CPU face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)

            cap = cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                logger.debug("Webcam could not be opened for gaze presence.")
                return

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.5)
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(60, 60))

                now = time.time()
                with self._lock:
                    if len(faces) > 0:
                        self._last_seen = now
                        if not self._user_present:
                            self._user_present = True
                            logger.info("User returned to desk.")
                            if self.on_presence_restored:
                                try: self.on_presence_restored()
                                except Exception: pass

                        # Check gaze engagement: larger centered face indicates looking directly at screen
                        (x, y, w, h) = faces[0]
                        frame_w = frame.shape[1]
                        center_x = x + (w / 2)
                        self._gaze_engaged = (abs(center_x - (frame_w / 2)) < (frame_w * 0.25)) and (w > 100)
                    else:
                        self._gaze_engaged = False
                        if self._user_present and (now - self._last_seen) > self.absence_timeout:
                            self._user_present = False
                            logger.info(f"User absent from desk for >{self.absence_timeout}s.")
                            if self.auto_lock_enabled and not self._locked:
                                self.lock_workstation()
                            if self.on_absence:
                                try: self.on_absence()
                                except Exception: pass

                # Check every 1.5 seconds for minimal CPU footprint
                time.sleep(1.5)

        except Exception as e:
            logger.debug(f"GazePresence monitor loop exception: {e}")
        finally:
            if cap:
                try: cap.release()
                except Exception: pass
