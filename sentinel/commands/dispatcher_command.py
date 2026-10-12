"""
sentinel/commands/dispatcher_command.py
────────────────────────────────────────
Voice Command Plugin for Autonomous Task Dispatcher & Multi-Device Mesh.

Handles: "start task dispatcher", "stop task dispatcher", "dispatcher status",
         "mesh devices", "dispatch task", "queue task", "cancel task", "register device".
"""

import logging
import re

from sentinel.core.task_dispatcher import (
    AutonomousTaskDispatcher,
)

logger = logging.getLogger("DispatcherCommand")


class DispatcherCommand:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.dispatcher = AutonomousTaskDispatcher(max_workers=4)

    def execute(self, command: str, entity: str | None = None) -> str:
        cmd = command.lower().strip()

        # 1. Start Dispatcher
        if any(
            k in cmd for k in ("start task dispatcher", "start dispatcher", "enable dispatcher", "launch dispatcher")
        ):
            started = self.dispatcher.start()
            if started:
                return "🚀 Autonomous Task Dispatcher started with 4 worker threads. Ready to route tasks across mesh nodes."
            return "Task Dispatcher is already running."

        # 2. Stop Dispatcher
        if any(
            k in cmd for k in ("stop task dispatcher", "stop dispatcher", "disable dispatcher", "shutdown dispatcher")
        ):
            self.dispatcher.stop()
            return "⏹️ Autonomous Task Dispatcher stopped. All active workers gracefully halted."

        # 3. Queue / Dispatcher Status
        if any(k in cmd for k in ("dispatcher status", "queue status", "task queue", "how many tasks", "mesh status")):
            status = self.dispatcher.get_queue_status()
            running_str = "Running" if status["running"] else "Stopped"
            return (
                f"📊 Task Dispatcher Status: {running_str}\n"
                f"• Workers: {status['workers']} | Queue Size: {status['queue_size']}\n"
                f"• Dispatched: {status['total_dispatched']} | Completed: {status['total_completed']} | Failed: {status['total_failed']}\n"
                f"• Mesh Devices: {status['mesh_devices']}"
            )

        # 4. List Mesh Devices
        if any(k in cmd for k in ("mesh devices", "list devices", "connected devices", "show devices", "device mesh")):
            devices = self.dispatcher.get_mesh_devices()
            if not devices:
                return "No devices registered in the mesh network."
            lines = ["🌐 Registered Mesh Devices:"]
            for d in devices:
                is_on = d.get("online", d.get("is_online", True))
                online_str = "🟢 Online" if is_on else "🔴 Offline"
                has_gpu = "gpu" in d.get("capabilities", []) or d.get("gpu", False)
                gpu_str = " (GPU)" if has_gpu else ""
                lines.append(
                    f"• {d['name']} [{d['type']}]{gpu_str} — {online_str} "
                    f"(Tasks: {d.get('tasks', 0)}, Load: {d.get('utilization', 0.0)}%)"
                )
            return "\n".join(lines)

        # 5. Cancel Task
        if any(k in cmd for k in ("cancel task", "abort task", "stop task")):
            task_id_match = re.search(r"(?:task\s+)?([a-f0-9\-]{8,12})", cmd)
            target_id = task_id_match.group(1) if task_id_match else (entity or "").strip()
            if target_id and self.dispatcher.cancel_task(target_id):
                return f"✅ Task '{target_id}' has been cancelled."
            return f"⚠️ Could not find or cancel task '{target_id}'."

        # 6. Task Status
        if any(k in cmd for k in ("task status", "check task")):
            task_id_match = re.search(r"(?:task\s+)?([a-f0-9\-]{8,12})", cmd)
            target_id = task_id_match.group(1) if task_id_match else (entity or "").strip()
            info = self.dispatcher.get_task_status(target_id) if target_id else None
            if info:
                return (
                    f"📋 Task '{info['name']}' ({info['task_id']}):\n"
                    f"• Status: {info['status']}\n"
                    f"• Device: {info['device'] or 'Unassigned'}\n"
                    f"• Retries: {info['retries']}" + (f"\n• Error: {info['error']}" if info["error"] else "")
                )
            return f"Task '{target_id}' not found."

        # 7. Register New Device
        if any(k in cmd for k in ("register device", "add device")):
            dev_name = entity or "remote_node"
            dev_id = self.dispatcher.register_device(
                name=dev_name,
                device_type="remote_server",
                capabilities=["cpu", "storage", "internet"],
            )
            return f"📱 Device '{dev_name}' registered to mesh with ID: {dev_id}."

        # 8. Submit / Dispatch Complex Task
        if any(k in cmd for k in ("dispatch task", "run task", "queue task", "submit task", "execute task")):
            task_desc = entity or cmd
            for prefix in ("dispatch task", "run task", "queue task", "submit task", "execute task"):
                if prefix in task_desc:
                    task_desc = task_desc.split(prefix, 1)[1].strip()

            if not task_desc:
                task_desc = "generic_task"

            # Auto start dispatcher if not already running
            if not self.dispatcher._running:
                self.dispatcher.start()

            subtasks = self.dispatcher.submit_complex_task(task_desc)
            if subtasks:
                sub_names = ", ".join(t.name for t in subtasks)
                return (
                    f"⚡ Complex task decomposed into {len(subtasks)} subtasks:\n"
                    f"• {sub_names}\n"
                    f"Dispatched across mesh nodes with dependency graph execution."
                )
            else:
                task = self.dispatcher.submit_task(name=task_desc)
                return f"⚡ Task '{task.name}' ({task.task_id}) submitted to priority queue."

        return (
            "Task Dispatcher commands:\n"
            "• 'start task dispatcher' / 'stop task dispatcher'\n"
            "• 'dispatcher status' / 'mesh devices'\n"
            "• 'dispatch task [description]' / 'cancel task [id]'"
        )
