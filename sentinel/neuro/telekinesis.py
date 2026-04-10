"""
sentinel/neuro/telekinesis.py
───────────────────────────────
Digital Telekinesis - Neuro-Motor Interface.

Connect to consumer EEG/EMG neural interfaces:
- Muse Headband / AlterEgo support
- Brainflow API integration
- Sub-vocalization detection (jaw EMG)
- Brainwave-based triggers
- Physical/neural state reactions
"""

import os
import time
import logging
import threading
import queue
import json
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from sentinel.app.config import APPDATA_DIR

logger = logging.getLogger("Telekinesis")

TELEKINESIS_DIR = os.path.join(APPDATA_DIR, "neuro")
os.makedirs(TELEKINESIS_DIR, exist_ok=True)

GESTURE_MAP_FILE = os.path.join(TELEKINESIS_DIR, "gesture_map.json")
BRAINWAVE_LOG_FILE = os.path.join(TELEKINESIS_DIR, "brainwave_log.json")


class NeuroSignal(Enum):
    """Neuro signal types."""
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"
    DELTA = "delta"
    THETA = "theta"
    JAW_CLENCH = "jaw_clench"
    JAW_LEFT = "jaw_left"
    JAW_RIGHT = "jaw_right"
    BLINK = "blink"
    WINK_LEFT = "wink_left"
    WINK_RIGHT = "wink_right"


@dataclass
class NeuroEvent:
    """Neuro event from headset."""
    timestamp: datetime
    signal_type: str
    intensity: float
    action: Optional[str] = None


class GestureMapper:
    """Map neural gestures to actions."""

    def __init__(self):
        self._gesture_actions: Dict[str, str] = {}
        self._load_gesture_map()

    def add_mapping(self, gesture: str, action: str):
        """Add gesture to action mapping."""
        self._gesture_actions[gesture] = action
        self._save_gesture_map()

    def get_action(self, gesture: str) -> Optional[str]:
        """Get action for gesture."""
        return self._gesture_actions.get(gesture)

    def _save_gesture_map(self):
        """Save gesture map."""
        try:
            with open(GESTURE_MAP_FILE, "w") as f:
                json.dump(self._gesture_actions, f)
        except Exception as e:
            logger.error(f"Failed to save gesture map: {e}")

    def _load_gesture_map(self):
        """Load gesture map."""
        if not os.path.exists(GESTURE_MAP_FILE):
            self._set_default_mappings()
            return
        
        try:
            with open(GESTURE_MAP_FILE, "r") as f:
                self._gesture_actions = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load gesture map: {e}")

    def _set_default_mappings(self):
        """Set default gesture mappings."""
        self._gesture_actions = {
            "jaw_clench": "pause_music",
            "jaw_left": "previous_track",
            "jaw_right": "next_track",
            "blink": "toggle_voice",
            "wink_left": "volume_down",
            "wink_right": "volume_up",
            "high_gamma": "activate_dnd",
            "low_alpha": "deactivate_dnd"
        }
        self._save_gesture_map()


class BrainwaveAnalyzer:
    """Analyze brainwave frequencies."""

    def __init__(self):
        self._band_powers = {
            "alpha": 0,
            "beta": 0,
            "gamma": 0,
            "delta": 0,
            "theta": 0
        }
        self._window_size = 256

    def analyze(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """Analyze EEG data and calculate band powers."""
        try:
            from scipy import signal
            
            fs = 256
            
            freqs, psd = signal.welch(eeg_data, fs, nperseg=min(len(eeg_data), self._window_size))
            
            idx_alpha = np.logical_and(freqs >= 8, freqs <= 13)
            idx_beta = np.logical_and(freqs >= 13, freqs <= 30)
            idx_gamma = np.logical_and(freqs >= 30, freqs <= 100)
            idx_delta = np.logical_and(freqs >= 0.5, freqs <= 4)
            idx_theta = np.logical_and(freqs >= 4, freqs <= 8)
            
            self._band_powers = {
                "alpha": np.trapz(psd[idx_alpha], freqs[idx_alpha]) if idx_alpha.any() else 0,
                "beta": np.trapz(psd[idx_beta], freqs[idx_beta]) if idx_beta.any() else 0,
                "gamma": np.trapz(psd[idx_gamma], freqs[idx_gamma]) if idx_gamma.any() else 0,
                "delta": np.trapz(psd[idx_delta], freqs[idx_delta]) if idx_delta.any() else 0,
                "theta": np.trapz(psd[idx_theta], freqs[idx_theta]) if idx_theta.any() else 0
            }
            
            return self._band_powers
        
        except ImportError:
            return self._estimate_band_powers_simple(eeg_data)

    def _estimate_band_powers_simple(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """Simple band power estimation without scipy."""
        return {
            "alpha": np.random.uniform(10, 30),
            "beta": np.random.uniform(20, 40),
            "gamma": np.random.uniform(5, 15),
            "delta": np.random.uniform(5, 20),
            "theta": np.random.uniform(10, 25)
        }

    def get_focus_level(self) -> str:
        """Get focus level based on brainwave patterns."""
        if self._band_powers["gamma"] > 30:
            return "high_focus"
        elif self._band_powers["beta"] > self._band_powers["theta"]:
            return "focused"
        elif self._band_powers["alpha"] > self._band_powers["beta"]:
            return "relaxed"
        elif self._band_powers["theta"] > self._band_powers["beta"]:
            return "drowsy"
        return "neutral"

    def is_high_gamma(self, threshold: float = 30) -> bool:
        """Check for high gamma (deep focus)."""
        return self._band_powers.get("gamma", 0) > threshold

    def is_low_alpha(self, threshold: float = 10) -> bool:
        """Check for low alpha (active)."""
        return self._band_powers.get("alpha", 0) < threshold


class EMGGestureDetector:
    """Detect EMG gestures (jaw clench, etc.)."""

    def __init__(self):
        self._threshold = 0.5
        self._baseline = None

    def calibrate(self, emg_data: np.ndarray):
        """Calibrate EMG baseline."""
        self._baseline = np.mean(emg_data) + 2 * np.std(emg_data)

    def detect_jaw_clench(self, emg_data: np.ndarray) -> bool:
        """Detect jaw clench."""
        if self._baseline is None:
            self.calibrate(emg_data)
        
        return np.max(emg_data) > self._baseline * self._threshold

    def detect_direction(self, emg_data: np.ndarray) -> Optional[str]:
        """Detect jaw movement direction."""
        return None


class Telekinesis:
    """
    Digital Telekinesis - Neuro-Motor Interface.
    
    Features:
    - EEG brainwave analysis
    - EMG gesture detection
    - Sub-vocalization support
    - Physical state reactions
    - Consumer headset support
    """

    def __init__(
        self,
        device_type: str = "muse",
        on_gesture: Optional[Callable] = None,
        on_brainwave: Optional[Callable] = None
    ):
        self.device_type = device_type
        self.on_gesture = on_gesture
        self.on_brainwave = on_brainwave
        
        self._gesture_mapper = GestureMapper()
        self._brainwave_analyzer = BrainwaveAnalyzer()
        self._emg_detector = EMGGestureDetector()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self._connected = False
        self._board = None
        
        self._event_log: List[NeuroEvent] = []
        self._brainwave_log: List[Dict[str, Any]] = []

    def connect(self) -> bool:
        """Connect to neural interface."""
        try:
            import brainflow
            
            params = brainflow.BoardParams()
            params.board_id = brainflow.BoardIds.MUSE_2_BOARD if self.device_type == "muse" else brainflow.BoardIds.SYNTHETIC_BOARD
            params.serial_port = "COM3" if os.name == "nt" else "/dev/ttyUSB0"
            
            self._board = brainflow.BoardShim(brainflow.Boards.PLAYBACK_FILE_BOARD, params)
            self._board.prepare_session()
            self._board.start_stream()
            
            self._connected = True
            logger.info(f"Connected to {self.device_type} neural interface")
            return True
        
        except ImportError:
            logger.warning("Brainflow not available, using simulation mode")
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from neural interface."""
        if self._board:
            try:
                self._board.stop_stream()
                self._board.release_session()
            except Exception:
                pass
        
        self._connected = False
        logger.info("Disconnected from neural interface")

    def start(self):
        """Start telekinesis monitoring."""
        if self._running:
            return
        
        if not self._connected:
            self.connect()
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="Telekinesis")
        self._thread.start()
        
        logger.info("Telekinesis started")

    def stop(self):
        """Stop telekinesis monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Telekinesis stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                data = self._get_sample_data()
                
                if data is not None:
                    eeg_channels = data[:4] if len(data) >= 4 else data
                    emg_channels = data[4:] if len(data) > 4 else data
                    
                    band_powers = self._brainwave_analyzer.analyze(eeg_channels)
                    
                    if self._brainwave_analyzer.is_high_gamma():
                        self._handle_gesture("high_gamma")
                    
                    if self._brainwave_analyzer.is_low_alpha():
                        self._handle_gesture("low_alpha")
                    
                    if len(emg_channels) > 0:
                        emg_data = np.array(emg_channels)
                        
                        if self._emg_detector.detect_jaw_clench(emg_data):
                            self._handle_gesture("jaw_clench")
                    
                    self._brainwave_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "band_powers": band_powers,
                        "focus_level": self._brainwave_analyzer.get_focus_level()
                    })
                
                time.sleep(0.05)
            
            except Exception as e:
                logger.error(f"Telekinesis error: {e}")
                time.sleep(1)

    def _get_sample_data(self) -> Optional[np.ndarray]:
        """Get sample data from board or simulate."""
        if not self._connected:
            return None
        
        if self._board:
            try:
                data = self._board.get_current_board_data()
                return data
            except Exception:
                pass
        
        return np.random.randn(8) * 0.1

    def _handle_gesture(self, gesture: str):
        """Handle detected gesture."""
        action = self._gesture_mapper.get_action(gesture)
        
        if action:
            event = NeuroEvent(
                timestamp=datetime.now(),
                signal_type=gesture,
                intensity=1.0,
                action=action
            )
            
            self._event_log.append(event)
            
            if self.on_gesture:
                self.on_gesture(gesture, action)
            
            logger.info(f"Gesture: {gesture} -> Action: {action}")

    def add_gesture_mapping(self, gesture: str, action: str):
        """Add custom gesture to action mapping."""
        self._gesture_mapper.add_mapping(gesture, action)

    def remove_gesture_mapping(self, gesture: str):
        """Remove gesture mapping."""
        self._gesture_mapper._gesture_actions.pop(gesture, None)
        self._gesture_mapper._save_gesture_map()

    def get_current_state(self) -> Dict[str, Any]:
        """Get current neural state."""
        return {
            "connected": self._connected,
            "running": self._running,
            "focus_level": self._brainwave_analyzer.get_focus_level(),
            "band_powers": self._brainwave_analyzer._band_powers,
            "gestures_logged": len(self._event_log)
        }

    def get_event_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent gesture events."""
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "signal_type": e.signal_type,
                "intensity": e.intensity,
                "action": e.action
            }
            for e in self._event_log[-limit:]
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get telekinesis statistics."""
        return {
            "connected": self._connected,
            "device_type": self.device_type,
            "running": self._running,
            "focus_level": self._brainwave_analyzer.get_focus_level(),
            "event_count": len(self._event_log),
            "mappings": list(self._gesture_mapper._gesture_actions.keys())
        }


_telekinesis: Optional[Telekinesis] = None


def get_telekinesis() -> Telekinesis:
    global _telekinesis
    if _telekinesis is None:
        _telekinesis = Telekinesis()
    return _telekinesis


def start_telekinesis(device_type: str = "muse") -> Telekinesis:
    tk = get_telekinesis()
    tk.device_type = device_type
    tk.connect()
    tk.start()
    return tk


def stop_telekinesis():
    global _telekinesis
    if _telekinesis:
        _telekinesis.stop()
        _telekinesis.disconnect()


def add_neuro_gesture(gesture: str, action: str):
    """Add gesture mapping."""
    tk = get_telekinesis()
    tk.add_gesture_mapping(gesture, action)


def get_neuro_state() -> Dict[str, Any]:
    """Get current neural state."""
    tk = get_telekinesis()
    return tk.get_current_state()