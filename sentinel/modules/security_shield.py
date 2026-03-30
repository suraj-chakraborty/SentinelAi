import psutil
import logging
import threading
import time
import os

class SecurityShieldModule:
    def __init__(self, notifier):
        self.notifier = notifier
        self.logger = logging.getLogger("SecurityShield")
        self.is_running = False
        self.known_partitions = set()
        self._initialize_partitions()

    def _initialize_partitions(self):
        """Records existing disk partitions to detect new USB insertions."""
        try:
            self.known_partitions = {p.device for p in psutil.disk_partitions()}
        except:
            pass

    def start_shield(self):
        """Starts the security monitoring loop."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._shield_loop, daemon=True)
        self.thread.start()
        self.logger.info("Sentinel Security Shield active.")

    def _shield_loop(self):
        while self.is_running:
            try:
                # 1. Detect New USB/Partitions
                current_partitions = {p.device for p in psutil.disk_partitions()}
                new_ones = current_partitions - self.known_partitions
                if new_ones:
                    msg = f"New hardware device detected: {', '.join(new_ones)}. Verify authorized use."
                    self.notifier.show_notification("Security Alert", msg)
                    self.logger.warning(msg)
                    self.known_partitions = current_partitions

                # 2. Monitor Network Spikes (Simple)
                net_io = psutil.net_io_counters()
                # If sending more than 5MB in a single minute loop (basic heuristic)
                if net_io.bytes_sent > 5 * 1024 * 1024: 
                    # Note: This would need a baseline to be truly effective
                    pass 

                # 3. Check for Suspicious Processes
                for proc in psutil.process_iter(['pid', 'name', 'username']):
                    if proc.info['name'] in ['cmd.exe', 'powershell.exe'] and proc.info['username'] != os.getlogin():
                        self.notifier.show_notification("Security Alert", f"Unauthorized terminal access detected (PID: {proc.info['pid']})")

            except Exception as e:
                self.logger.error(f"Security shield error: {e}")
            
            time.sleep(10) # High frequency check for security

    def stop_shield(self):
        self.is_running = False
