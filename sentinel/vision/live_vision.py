"""
sentinel/vision/live_vision.py
──────────────────────────────
Live spatial vision module - webcam streaming and environmental awareness.
"""

import os
import time
import logging
import threading
import queue
import base64
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import cv2

from sentinel.app.config import APPDATA_DIR

logger = logging.getLogger("LiveVision")

VIDEOS_DIR = os.path.join(APPDATA_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)


@dataclass
class VisionFrame:
    """Represents a captured frame with metadata."""
    frame: np.ndarray
    timestamp: datetime
    width: int
    height: int
    camera_id: int


class LiveVisionStream:
    """Live webcam stream for environmental awareness."""

    def __init__(
        self,
        camera_id: int = 0,
        fps: int = 10,
        resolution: tuple = (640, 480),
        on_frame: Optional[Callable] = None
    ):
        self.camera_id = camera_id
        self.fps = fps
        self.resolution = resolution
        self.on_frame = on_frame
        
        self._capture = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_queue = queue.Queue(maxsize=10)
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_count = 0
        self._start_time: Optional[datetime] = None

    def start(self) -> bool:
        """Start the video capture stream."""
        if self._running:
            logger.warning("Vision stream already running")
            return True
        
        try:
            self._capture = cv2.VideoCapture(self.camera_id)
            if not self._capture.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            self._running = True
            self._start_time = datetime.now()
            self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="LiveVision")
            self._thread.start()
            
            logger.info(f"Live vision started (camera={self.camera_id}, fps={self.fps})")
            return True
        except Exception as e:
            logger.error(f"Failed to start vision stream: {e}")
            return False

    def stop(self):
        """Stop the video capture stream."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._capture:
            self._capture.release()
            self._capture = None
        logger.info("Live vision stopped")

    def _capture_loop(self):
        """Main capture loop running in background thread."""
        while self._running:
            try:
                if self._capture and self._capture.isOpened():
                    ret, frame = self._capture.read()
                    if ret:
                        self._latest_frame = frame
                        self._frame_count += 1
                        
                        vision_frame = VisionFrame(
                            frame=frame,
                            timestamp=datetime.now(),
                            width=frame.shape[1],
                            height=frame.shape[0],
                            camera_id=self.camera_id
                        )
                        
                        try:
                            self._frame_queue.put_nowait(vision_frame)
                        except queue.Full:
                            try:
                                self._frame_queue.get_nowait()
                                self._frame_queue.put_nowait(vision_frame)
                            except queue.Empty:
                                pass
                        
                        if self.on_frame:
                            self.on_frame(frame)
                
                time.sleep(1.0 / self.fps)
            except Exception as e:
                logger.error(f"Vision capture error: {e}")
                time.sleep(1)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame."""
        return self._latest_frame

    def get_frame_base64(self) -> Optional[str]:
        """Get latest frame as base64 JPEG."""
        frame = self._latest_frame
        if frame is None:
            return None
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Frame encoding failed: {e}")
            return None

    def capture_frame(self) -> Optional[str]:
        """Capture and save current frame to file."""
        frame = self._latest_frame
        if frame is None:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vision_{timestamp}.jpg"
        filepath = os.path.join(VIDEOS_DIR, filename)
        
        try:
            cv2.imwrite(filepath, frame)
            logger.info(f"Frame saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Frame save failed: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        duration = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
        return {
            "running": self._running,
            "camera_id": self.camera_id,
            "fps": self.fps,
            "resolution": self.resolution,
            "frame_count": self._frame_count,
            "duration_seconds": duration,
            "avg_fps": self._frame_count / duration if duration > 0 else 0
        }

    def analyze_current_frame(self, prompt: str = "Describe what you see in this image.") -> str:
        """Analyze the current frame using vision AI."""
        frame = self._latest_frame
        if frame is None:
            return "No frame available for analysis"
        
        try:
            from sentinel.modules.vision import VisionModule
            vision = VisionModule()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_path = os.path.join(VIDEOS_DIR, f"temp_{timestamp}.jpg")
            cv2.imwrite(temp_path, frame)
            
            result = vision.analyze_screen(prompt)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            return f"Analysis failed: {e}"


_live_stream: Optional[LiveVisionStream] = None


def get_live_vision(camera_id: int = 0) -> LiveVisionStream:
    global _live_stream
    if _live_stream is None:
        _live_stream = LiveVisionStream(camera_id=camera_id)
    return _live_stream


def start_vision_stream(camera_id: int = 0, fps: int = 10) -> bool:
    stream = get_live_vision(camera_id)
    stream.fps = fps
    return stream.start()


def stop_vision_stream():
    global _live_stream
    if _live_stream:
        _live_stream.stop()


def get_vision_frame() -> Optional[str]:
    stream = get_live_vision()
    return stream.get_frame_base64()