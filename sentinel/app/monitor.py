"""
sentinel/app/monitor.py
───────────────────────
System Health & Performance Monitoring.
Tracks LLM latency, memory usage, and plugin status.
"""

import time
import threading
import logging
import psutil
from typing import Dict

from sentinel.app.state import get_state

logger = logging.getLogger("SentinelMonitor")

class HealthMonitor:
    """
    Background monitor to update system health metrics.
    """
    
    def __init__(self, interval: int = 30):
        self.state = get_state()
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("Health Monitoring Service Started.")

    def stop(self):
        self.running = False

    def _run(self):
        while self.running:
            try:
                self._update_system_metrics()
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            time.sleep(self.interval)

    def _update_system_metrics(self):
        # 1. CPU/Memory
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        self.state.set_status("SYS_CPU", cpu)
        self.state.set_status("SYS_MEM", mem)
        
        # 2. Process Stats
        proc = psutil.Process()
        proc_mem = proc.memory_info().rss / (1024 * 1024) # MB
        self.state.set_status("SENTINEL_MEM_MB", int(proc_mem))
        
        # 3. Connectivity (Placeholder for latency tracking)
        # In a real scenario, this would be updated by the intelligence module
        # during each API call.
        
        logger.debug(f"Metrics Updated: CPU {cpu}%, Mem {mem}%")

    def record_latency(self, service: str, ms: float):
        """Record the latency of an external service call."""
        # Moving window average or just latest
        self.state.set_status(f"LATENCY_{service.upper()}", ms)
        
        if ms > 2000: # Over 2s
            self.state.set_status("STABILITY_WARNING", True)
        else:
            self.state.set_status("STABILITY_WARNING", False)
