import psutil
import logging
try:
    import GPUtil
except ImportError:
    GPUtil = None

class SystemMonitor:
    def __init__(self, temp_threshold=80):
        self.temp_threshold = temp_threshold
        self.logger = logging.getLogger("SystemMonitor")

    def get_cpu_temp(self) -> float:
        """Attempt to read CPU temperature, handling Windows/psutil limitations."""
        if not hasattr(psutil, "sensors_temperatures"):
            return None
            
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return None
            for name, entries in temps.items():
                for entry in entries:
                    return entry.current
        except Exception:
            return None
        return None

    def get_gpu_temp(self):
        if not GPUtil:
            return None
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].temperature
        except Exception as e:
            self.logger.error(f"Error getting GPU temp: {e}")
            return None
        return None

    def check_temp_alerts(self):
        cpu_temp = self.get_cpu_temp()
        gpu_temp = self.get_gpu_temp()
        alerts = []
        if cpu_temp and cpu_temp > self.temp_threshold:
            alerts.append(f"CPU temperature high: {cpu_temp}°C")
        if gpu_temp and gpu_temp > self.temp_threshold:
            alerts.append(f"GPU temperature high: {gpu_temp}°C")
        return alerts

    def check_misuse(self):
        """Identify processes with abnormally high CPU usage."""
        misuse_alerts = []
        try:
            # We use a short interval for process_iter to get real-time CPU %
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    # Ignore the 'System Idle Process' and 'System'
                    if (proc.info['name'] or "").lower() in ('system idle process', 'system', 'idle'):
                        continue
                    if proc.info['cpu_percent'] > 95:
                        misuse_alerts.append(f"Process {proc.info['name']} (PID: {proc.info['pid']}) is using high CPU: {proc.info['cpu_percent']}%")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        return misuse_alerts
