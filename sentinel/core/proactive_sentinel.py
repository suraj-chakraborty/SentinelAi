"""
sentinel/core/proactive_sentinel.py
───────────────────────────────────
Proactive Sentinel Daemon.
Monitors system vitals, battery, and processes autonomously,
speaking proactive warnings before critical issues impact the user.
"""

import threading
import time
import logging
from typing import Optional, Dict

logger = logging.getLogger("ProactiveSentinel")

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


class ProactiveSentinel:
    """
    Autonomous background watcher.
    Alerts on high CPU saturation, low battery, or low disk space.
    """

    def __init__(self, orchestrator, check_interval: int = 60):
        self.orchestrator = orchestrator
        self.check_interval = check_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Tracking state & cooldowns (seconds)
        self._last_cpu_alert = 0.0
        self._last_battery_alert = 0.0
        self._last_disk_alert = 0.0
        self._high_cpu_streak = 0
        self._cooldown = 900.0  # 15 minutes between repeated alerts of same type

    def start(self):
        """Start background daemon."""
        if self._running or not _PSUTIL_AVAILABLE:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ProactiveSentinelLoop")
        self._thread.start()
        logger.info("ProactiveSentinel daemon started.")

    def stop(self):
        """Stop background daemon."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("ProactiveSentinel daemon stopped.")

    def _speak_alert(self, message: str):
        """Vocalize alert via orchestrator speak_fn or notifier."""
        logger.warning(f"[PROACTIVE ALERT] {message}")
        try:
            if hasattr(self.orchestrator, "speak_fn") and self.orchestrator.speak_fn:
                self.orchestrator.speak_fn(message)
            elif hasattr(self.orchestrator, "notifier") and self.orchestrator.notifier:
                self.orchestrator.notifier.notify("Sentinel Alert", message)
        except Exception as e:
            logger.debug(f"Failed to deliver proactive alert: {e}")

    def _run_loop(self):
        # Initial warmup delay so startup isn't interrupted
        time.sleep(15)

        while self._running:
            try:
                now = time.time()

                # 1. CPU Saturation Check
                cpu_pct = psutil.cpu_percent(interval=1.0)
                if cpu_pct >= 90.0:
                    self._high_cpu_streak += 1
                    if self._high_cpu_streak >= 2 and (now - self._last_cpu_alert) > self._cooldown:
                        # Find top process
                        top_proc = "a background task"
                        try:
                            procs = sorted(
                                psutil.process_iter(['name', 'cpu_percent']),
                                key=lambda p: p.info['cpu_percent'] or 0,
                                reverse=True
                            )
                            if procs and procs[0].info['name']:
                                top_proc = procs[0].info['name']
                        except Exception:
                            pass

                        self._speak_alert(f"Pardon the interruption, sir. CPU usage is peaking at {int(cpu_pct)} percent due to {top_proc}.")
                        self._last_cpu_alert = now
                        self._high_cpu_streak = 0
                else:
                    self._high_cpu_streak = 0

                # 2. Battery Check
                battery = psutil.sensors_battery()
                if battery is not None:
                    if not battery.power_plugged and battery.percent <= 15:
                        if (now - self._last_battery_alert) > self._cooldown:
                            self._speak_alert(f"Battery is down to {int(battery.percent)} percent. Please connect your charger.")
                            self._last_battery_alert = now

                # 3. Disk Space Check
                try:
                    disk = psutil.disk_usage("C:\\")
                    free_gb = disk.free / (1024 ** 3)
                    if free_gb < 4.0 and (now - self._last_disk_alert) > (self._cooldown * 2):
                        self._speak_alert(f"Warning: Primary drive has less than {int(free_gb)} gigabytes of free storage remaining.")
                        self._last_disk_alert = now
                except Exception:
                    pass

            except Exception as e:
                logger.debug(f"Proactive loop iteration error: {e}")

            time.sleep(self.check_interval)
