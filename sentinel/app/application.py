"""
sentinel/app/application.py
──────────────────────────
The main Sentinel Application class that coordinates the GUI, 
audio listeners, and background orchestration.
"""

import threading
import queue
import os
import sys
import time
import datetime
import logging
import warnings
import pvporcupine
import pyaudio
import numpy as np

# ── Noise Filter ─────────────────────────────────────────────────────────────
class NoiseFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
    def write(self, s):
        try:
            # Handle both string and bytes gracefully
            if isinstance(s, bytes):
                msg = s.decode('utf-8', errors='ignore')
            else:
                msg = str(s)
                
            # Filter Windows-specific chatter
            if any(k in msg for k in ("WNDPROC", "WPARAM is simple", "LRESULT")):
                return len(s)
            
            # Use original_stderr safely
            self.original_stderr.write(msg)
            return len(s)
        except Exception:
            # Absolute fallback to avoid recursion or stream corruption
            try:
                self.original_stderr.write(str(s))
                return len(s)
            except:
                return len(s)

    def flush(self):
        try:
            self.original_stderr.flush()
        except:
            pass

if sys.platform == "win32":
    sys.stderr = NoiseFilter(sys.stderr)

from sentinel.app.state import get_state
from sentinel.app.config import (
    ACCESS_KEY, COMMAND_WAV, PORCUPINE_KEYWORD_PATH,
    get_porcupine_model_path, load_settings
)
from sentinel.app.gui import build_dashboard, RecordingOverlay, start_tray_icon
from sentinel.app.audio import record_until_silence, liveness_verify
from sentinel.app.voice import speak, transcribe
from sentinel.app.orchestration import execute_command, start_alert_loop

# Suppress harmless but noisy WNDPROC warnings on Windows 11
warnings.filterwarnings("ignore", message="WPARAM is simple, so must be an int object")

logger = logging.getLogger("SentinelApp")

class SentinelApp:
    def __init__(self):
        self.state = get_state()
        self.status_queue = queue.Queue()
        self.orchestrator = None
        self.root = None
        self.overlay = None
        self.wake_stop_event = threading.Event()
        
    def log_status(self, msg: str):
        """Send a message to the GUI status bar."""
        self.status_queue.put(msg)
        logger.info(f"[Status] {msg}")

    def on_amp_update(self, amp: float):
        """Callback for the recording overlay."""
        if self.overlay:
            self.overlay.update_amp(amp)

    # ── Wake Word Listener ───────────────────────────────────────────────────

    def _listen_loop(self):
        """Background thread listening for the wake word."""
        if not ACCESS_KEY:
            self.log_status("Porcupine Key Missing.")
            return

        settings = load_settings()
        # Ensure we don't use an empty string if the setting exists but is blank
        custom_path = settings.get("wake_word_path") or PORCUPINE_KEYWORD_PATH
        sens = float(settings.get("wake_sensitivity", 0.5))
        model_path = get_porcupine_model_path()
        
        try:
            if custom_path and os.path.exists(custom_path):
                logger.info(f"Initializing wake word with custom model: {os.path.basename(custom_path)}")
                porcupine = pvporcupine.create(
                    access_key=ACCESS_KEY,
                    keyword_paths=[custom_path],
                    sensitivities=[sens],
                    model_path=model_path
                )
            else:
                logger.info("Initializing wake word with default keyword: 'computer'")
                porcupine = pvporcupine.create(
                    access_key=ACCESS_KEY,
                    keywords=["computer"],
                    sensitivities=[sens],
                    model_path=model_path
                )
        except Exception as e:
            self.log_status(f"Wake word init failed: {str(e)[:20]}")
            return

        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(format=pyaudio.paInt16, channels=1, rate=porcupine.sample_rate,
                             input=True, frames_per_buffer=porcupine.frame_length)
        except Exception as e:
            msg = f"Mic open failed: {str(e)[:30]}"
            self.log_status(msg)
            logger.error(msg)
            pa.terminate()
            porcupine.delete()
            return

        self.log_status("Listening for wake word...")
        self._is_paused = False
        loop_counter = 0
        
        try:
            while not self.wake_stop_event.is_set():
                if self._is_paused:
                    time.sleep(0.1)
                    continue
                    
                try:
                    pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
                    pcm_unpacked = np.frombuffer(pcm, dtype=np.int16)
                    
                    # Update real-time voice meter
                    amp = np.abs(pcm_unpacked).astype(float).mean()
                    meter_sens = settings.get("meter_sensitivity", 50)
                    meter_val = min(100, int((amp / 1000) * meter_sens))
                    self.root.after_idle(lambda v=meter_val: self.voice_meter.configure(value=v))

                    # Heartbeat log every ~3 seconds
                    loop_counter += 1
                    if loop_counter % 100 == 0:
                        logger.debug(f"Audio heartbeat - Avg Amp: {amp:.2f}")

                    if porcupine.process(pcm_unpacked) >= 0:
                        logger.info("Wake word detected!")
                        self._is_paused = True
                        threading.Thread(target=self._handle_wake_detection, daemon=True).start()
                except Exception as e:
                    logger.debug(f"Stream read error: {e}")
                    time.sleep(0.1)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            porcupine.delete()

    def _is_session_active(self) -> bool:
        """Check if the user has authenticated since the most recent 12:30 AM."""
        last_auth = self.state.last_auth_time
        if last_auth <= 0:
            return False
            
        now = datetime.datetime.now()
        # Find the most recent 12:30 AM
        reset_time = now.replace(hour=0, minute=30, second=0, microsecond=0)
        if now < reset_time:
            # If current time is before 12:30 AM, the reset was yesterday at 12:30 AM
            reset_time -= datetime.timedelta(days=1)
            
        return last_auth > reset_time.timestamp()

    def _verify_identification_code(self) -> bool:
        """Prompt for and verify a spoken 4-digit identification code."""
        speak("Identification required for new session. Please state your four digit code.", block=True)
        self.log_status("Waiting for ID code...")
        
        # Open overlay for visual feedback
        self.overlay = RecordingOverlay(self.root)
        fn, had = record_until_silence(COMMAND_WAV, on_amp=self.on_amp_update, max_duration=6)
        self.overlay.destroy()
        self.overlay = None
        
        if not had:
            speak("No code detected. Access denied.")
            return False
            
        code_text = transcribe(fn)
        if not code_text:
            speak("I couldn't hear the code. Access denied.")
            return False
            
        # Clean up text (remove spaces, etc)
        digits = "".join(filter(str.isdigit, code_text))
        
        # For prototype, we use '1234' or fetch from secrets
        from sentinel.app.secrets import fetch_secret
        # We need a master password to fetch, but for session wake we use a hardcoded or env-based ID for now
        # unless we want to ask for password (the user said 'identification code')
        target_code = os.getenv("SENTINEL_ID_CODE", "1234")
        
        if digits == target_code:
            self.state.last_auth_time = time.time()
            speak("Identity confirmed. Session active until 12 30 AM.")
            self.log_status("Session Activated")
            return True
        else:
            logger.warning(f"Failed ID attempt: {digits}")
            speak("Incorrect code. Access denied.")
            return False

    def _handle_wake_detection(self):
        """Handle wake word event: Session Check -> Record -> Transcribe -> Execute."""
        try:
            if self.state.get_status("LISTENING_PAUSED"):
                return

            self.log_status("Wake word detected.")
            
            # 1. Session Check
            if not self._is_session_active():
                if not self._verify_identification_code():
                    return
            else:
                # Subtle chime or "Yes" if already auth'd
                speak("Yes Master", block=True)
            
            # 2. Record command
            self.log_status("Recording command...")
            self.overlay = RecordingOverlay(self.root)
            fn, had = record_until_silence(COMMAND_WAV, on_amp=self.on_amp_update)
            self.overlay.destroy()
            self.overlay = None
            
            if not had:
                self.log_status("No speech detected.")
                return

            # 3. Process
            self.log_status("Processing...")
            text = transcribe(fn)
            if text:
                print(f"\n[USER COMMAND]: {text}")
                self.log_status(f"Command: {text}")
                execute_command(text, self.orchestrator)
            else:
                speak("I'm sorry, I couldn't understand.")
        finally:
            self._is_paused = False

    def _on_exit(self):
        """Callback to shut down the entire app gracefully."""
        self.log_status("Shutting down...")
        if self.root:
            self.root.after(100, self.root.destroy)
        else:
            os._exit(0)

    # ── Application Lifecycle ────────────────────────────────────────────────

    def run(self):
        """Start all services and show GUI."""
        from sentinel.app.boot import initialize_orchestrator
        self.orchestrator = initialize_orchestrator()
        
        if self.orchestrator:
            self.orchestrator.exit_callback = self._on_exit # Override with GUI closer
            start_alert_loop(self.orchestrator)

        # 1. Startup Greeting
        speak("Sentinel OS V4 online. How can I assist?", block=False)

        # 2. Start wake listener
        threading.Thread(target=self._listen_loop, daemon=True).start()

        # Build GUI
        from sentinel.app.voice_enroll import start_enrollment
        from sentinel.app.secrets import verify_master
        from tkinter import simpledialog

        def on_voice_reg_click():
            pwd = simpledialog.askstring("Master Password", "Enter Master Password to recalibrate voice:", show='*')
            if verify_master(pwd):
                threading.Thread(target=start_enrollment, args=(self.log_status,), daemon=True).start()
            else:
                self.log_status("Authentication failed.")

        self.root, self.voice_meter, self.mru_combo = build_dashboard(
            on_voice_reg=on_voice_reg_click,
            on_rebuild=lambda: threading.Thread(target=build_app_registry, kwargs={"fast_only": False}, daemon=True).start(),
            on_manual=lambda: self._handle_wake_detection(), # Reuse wake logic
            on_config=lambda: print("Show Config"),
            on_terminate=lambda: self._on_exit(),
            on_agent_toggle=lambda: print("Agent Toggle"),
            app_list=[] # Load from registry
        )

        def update_ui():
            while not self.status_queue.empty():
                msg = self.status_queue.get_nowait()
                # Update status label in root (needs a reference, simplified here)
            self.root.after(400, update_ui)

        self.root.after(400, update_ui)
        self.root.mainloop()
        self.wake_stop_event.set()
