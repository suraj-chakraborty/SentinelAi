import logging
import os
import subprocess

logger = logging.getLogger("SystemControl")

class SystemControl:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def execute(self, command: str, entity: str) -> str:
        """
        Logic for system controls: shutdown, restart, status.
        """
        if "shutdown" in command:
            return self.shutdown()
        elif "restart" in command or "reboot" in command:
            return self.restart()
        elif "status" in command or "info" in command:
            return self.system_info()
            
        return "Unknown system command."

    def shutdown(self) -> str:
        try:
            os.system("shutdown /s /t 60")
            return "System shutting down in 60 seconds. Say 'abort shutdown' if you change your mind."
        except Exception as e:
            return f"Failed to shutdown: {e}"

    def restart(self) -> str:
        try:
            os.system("shutdown /r /t 60")
            return "System restarting in 60 seconds."
        except Exception as e:
            return f"Failed to restart: {e}"

    def system_info(self) -> str:
        if self.orchestrator and hasattr(self.orchestrator, "system_monitor"):
            monitor = self.orchestrator.system_monitor
            cpu_temp = monitor.get_cpu_temp()
            gpu_temp = monitor.get_gpu_temp()
            return f"System Status: CPU Temp: {cpu_temp if cpu_temp else 'N/A'}°C, GPU Temp: {gpu_temp if gpu_temp else 'N/A'}°C"
        return "System monitoring module is unavailable."
