"""
sentinel/commands/npu_command.py
─────────────────────────────────
Voice Command Plugin for NPU/GPU Neural Acceleration.

Handles: "npu status", "gpu status", "accelerator info",
         "start gpu monitoring", "stop gpu monitoring",
         "recommend quantization".
"""

import logging
from typing import Optional

from sentinel.neuro.npu_accelerator import NeuralAccelerationEngine

logger = logging.getLogger("NPUCommand")


class NPUCommand:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.engine = NeuralAccelerationEngine()

    def execute(self, command: str, entity: Optional[str] = None) -> str:
        cmd = command.lower().strip()

        # 1. Initialize / detect
        if any(k in cmd for k in ("detect accelerator", "initialize npu", "detect gpu",
                                   "scan hardware", "npu init")):
            result = self.engine.initialize()
            devices = [f"  • {d['backend']}: {d['device']}" for d in result["devices"]]
            return (
                f"⚡ Neural acceleration initialized!\n"
                f"Backends detected: {result['backends_available']}\n"
                f"Primary: {result['primary']}\n" +
                "\n".join(devices)
            )

        # 2. Status
        if any(k in cmd for k in ("npu status", "gpu status", "accelerator status",
                                   "accelerator info", "neural status")):
            if not self.engine._initialized:
                self.engine.initialize()
            status = self.engine.get_status()
            lines = [f"Neural Accelerator: {'Initialized' if status['initialized'] else 'Not initialized'}"]
            for b in status["backends"]:
                lines.append(f"  • {b['backend']}: {b['device']} ({'✓' if b['available'] else '✗'})")
            if status["monitoring"]:
                lines.append(f"GPU: {status['gpu_utilization']:.0f}% util, {status['gpu_temperature']:.0f}°C")
                if status["thermal_throttle"]:
                    lines.append("⚠️ THERMAL THROTTLE ACTIVE")
            return "\n".join(lines)

        # 3. Start monitoring
        if any(k in cmd for k in ("start gpu monitor", "monitor gpu", "start npu monitor",
                                   "track gpu")):
            if not self.engine._initialized:
                self.engine.initialize()
            self.engine.start_monitoring()
            return "📊 GPU/NPU monitoring started. Real-time thermal and utilization tracking active."

        # 4. Stop monitoring
        if any(k in cmd for k in ("stop gpu monitor", "stop npu monitor")):
            self.engine.stop_monitoring()
            return "GPU/NPU monitoring stopped."

        # 5. GPU temperature / stats
        if any(k in cmd for k in ("gpu temperature", "gpu temp", "how hot is gpu",
                                   "gpu utilization")):
            stats = self.engine.get_gpu_stats()
            return (
                f"GPU Stats:\n"
                f"  Utilization: {stats['utilization_pct']:.0f}%\n"
                f"  Temperature: {stats['temperature_c']:.0f}°C\n"
                f"  Throttle: {'⚠️ YES' if stats['thermal_throttle'] else '✅ No'} "
                f"(limit: {stats['thermal_limit_c']}°C)"
            )

        # 6. Performance summary
        if any(k in cmd for k in ("inference performance", "npu performance",
                                   "acceleration stats")):
            perf = self.engine.get_performance_summary()
            if "message" in perf:
                return perf["message"]
            return (
                f"Performance Summary:\n"
                f"  Total inferences: {perf['total_inferences']}\n"
                f"  Avg latency: {perf['avg_latency_ms']:.1f}ms\n"
                f"  Backend usage: {perf['backend_distribution']}"
            )

        return "NPU command not recognized. Try: npu status, detect accelerator, start gpu monitoring."
