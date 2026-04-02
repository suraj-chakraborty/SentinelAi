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

    def get_cpu_temp(self):
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return None
            for name, entries in temps.items():
                for entry in entries:
                    return entry.current
        except Exception as e:
            self.logger.error(f"Error getting CPU temp: {e}")
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
        # Basic misuse check: High CPU/RAM processes not recognized
        misuse_alerts = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if proc.info['cpu_percent'] > 90:
                    misuse_alerts.append(f"Process {proc.info['name']} (PID: {proc.info['pid']}) is using high CPU: {proc.info['cpu_percent']}%")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return misuse_alerts
