"""
sentinel/commands/daemon_control.py
───────────────────────────────────
Command to expose daemon management capabilities.
"""

import logging
from typing import Dict, Any, Optional

from sentinel.daemons.daemon_manager import get_daemon_manager, start_daemon_manager, stop_daemon_manager

logger = logging.getLogger("DaemonControlCommand")


class DaemonControlCommand:
    def __init__(self):
        pass

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute a daemon control action."""
        try:
            manager = get_daemon_manager()
            
            if action == "start":
                start_daemon_manager()
                return {"success": True, "message": "Daemon manager started"}
            
            elif action == "stop":
                stop_daemon_manager()
                return {"success": True, "message": "Daemon manager stopped"}
            
            elif action == "list":
                return {"success": True, "tasks": manager.list_tasks()}
            
            elif action == "run":
                task_id = kwargs.get("task_id")
                if not task_id:
                    return {"success": False, "error": "task_id required"}
                return manager.run_task_now(task_id)
            
            elif action == "status":
                task_id = kwargs.get("task_id")
                if not task_id:
                    return {"success": False, "error": "task_id required"}
                status = manager.get_task_status(task_id)
                if status:
                    return {"success": True, "status": status}
                return {"success": False, "error": f"Task {task_id} not found"}
            
            elif action == "enable":
                task_id = kwargs.get("task_id")
                if not task_id:
                    return {"success": False, "error": "task_id required"}
                if manager.enable_task(task_id):
                    return {"success": True, "message": f"Task {task_id} enabled"}
                return {"success": False, "error": "Task not found"}
            
            elif action == "disable":
                task_id = kwargs.get("task_id")
                if not task_id:
                    return {"success": False, "error": "task_id required"}
                if manager.disable_task(task_id):
                    return {"success": True, "message": f"Task {task_id} disabled"}
                return {"success": False, "error": "Task not found"}
            
            elif action == "stats":
                return manager.get_stats()
            
            elif action == "events":
                return {"success": True, "events": manager.get_events()}
            
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        
        except Exception as e:
            logger.error(f"Daemon control failed: {e}")
            return {"success": False, "error": str(e)}


_daemon_command: Optional[DaemonControlCommand] = None


def get_daemon_command() -> DaemonControlCommand:
    global _daemon_command
    if _daemon_command is None:
        _daemon_command = DaemonControlCommand()
    return _daemon_command