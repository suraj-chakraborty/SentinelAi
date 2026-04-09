"""
sentinel/daemons/daemon_manager.py
──────────────────────────────────
Context-aware background daemon manager.
Schedules and runs background tasks with triggers and context awareness.
"""

import os
import time
import logging
import threading
import queue
import json
import schedule
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from enum import Enum

from sentinel.app.config import APPDATA_DIR

logger = logging.getLogger("DaemonManager")

DAEMON_STATE_FILE = os.path.join(APPDATA_DIR, "daemon_state.json")


class DaemonStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class DaemonTask:
    """Represents a background task."""
    id: str
    name: str
    description: str
    action: Callable
    trigger: str
    interval_seconds: int = 0
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: DaemonStatus = DaemonStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None


class DaemonManager:
    """Manages background daemons and scheduled tasks."""

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._tasks: Dict[str, DaemonTask] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._event_queue = queue.Queue()
        self._schedule_thread: Optional[threading.Thread] = None

    def register_task(
        self,
        task_id: str,
        name: str,
        action: Callable,
        trigger: str = "manual",
        interval_seconds: int = 0,
        description: str = ""
    ) -> bool:
        """Register a new background task."""
        try:
            task = DaemonTask(
                id=task_id,
                name=name,
                description=description,
                action=action,
                trigger=trigger,
                interval_seconds=interval_seconds
            )
            self._tasks[task_id] = task
            
            if trigger == "interval" and interval_seconds > 0:
                schedule.every(interval_seconds).seconds.do(self._run_task, task_id)
            
            logger.info(f"Registered daemon task: {task_id} (trigger: {trigger})")
            return True
        except Exception as e:
            logger.error(f"Failed to register task {task_id}: {e}")
            return False

    def unregister_task(self, task_id: str) -> bool:
        """Unregister a task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False

    def start(self):
        """Start all enabled daemons."""
        if self._running:
            logger.warning("Daemon manager already running")
            return
        
        self._running = True
        
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="DaemonManager")
        self._thread.start()
        
        self._schedule_thread = threading.Thread(target=self._schedule_loop, daemon=True, name="ScheduleRunner")
        self._schedule_thread.start()
        
        logger.info("Daemon manager started")

    def stop(self):
        """Stop all daemons."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._schedule_thread:
            self._schedule_thread.join(timeout=5)
        logger.info("Daemon manager stopped")

    def _run_loop(self):
        """Main daemon loop."""
        while self._running:
            try:
                for task_id, task in self._tasks.items():
                    if task.enabled and task.trigger == "always":
                        self._run_task(task_id)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Daemon loop error: {e}")
                time.sleep(5)

    def _schedule_loop(self):
        """Run scheduled tasks."""
        while self._running:
            try:
                schedule.run_pending()
            except Exception as e:
                logger.error(f"Schedule loop error: {e}")
            time.sleep(1)

    def _run_task(self, task_id: str):
        """Execute a task."""
        if task_id not in self._tasks:
            return
        
        task = self._tasks[task_id]
        task.status = DaemonStatus.RUNNING
        task.last_run = datetime.now()
        
        try:
            logger.info(f"Running daemon task: {task_id}")
            result = task.action()
            task.result = str(result) if result else "Completed"
            task.status = DaemonStatus.COMPLETED
            
            self._event_queue.put({
                "type": "task_completed",
                "task_id": task_id,
                "timestamp": task.last_run.isoformat()
            })
        except Exception as e:
            task.error = str(e)
            task.status = DaemonStatus.FAILED
            logger.error(f"Task {task_id} failed: {e}")
            
            self._event_queue.put({
                "type": "task_failed",
                "task_id": task_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    def run_task_now(self, task_id: str) -> Dict[str, Any]:
        """Manually trigger a task."""
        if task_id not in self._tasks:
            return {"success": False, "error": f"Task {task_id} not found"}
        
        self._run_task(task_id)
        return {"success": True, "task_id": task_id, "status": self._tasks[task_id].status.value}

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task."""
        if task_id not in self._tasks:
            return None
        task = self._tasks[task_id]
        return {
            "id": task.id,
            "name": task.name,
            "status": task.status.value,
            "last_run": task.last_run.isoformat() if task.last_run else None,
            "next_run": task.next_run.isoformat() if task.next_run else None,
            "result": task.result,
            "error": task.error
        }

    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all registered tasks."""
        return [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "trigger": t.trigger,
                "interval_seconds": t.interval_seconds,
                "enabled": t.enabled,
                "status": t.status.value
            }
            for t in self._tasks.values()
        ]

    def enable_task(self, task_id: str) -> bool:
        """Enable a task."""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = True
            return True
        return False

    def disable_task(self, task_id: str) -> bool:
        """Disable a task."""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = False
            return True
        return False

    def get_events(self, timeout: float = 0.1) -> List[Dict[str, Any]]:
        """Get recent daemon events."""
        events = []
        while True:
            try:
                event = self._event_queue.get_nowait()
                events.append(event)
            except queue.Empty:
                break
        return events

    def get_stats(self) -> Dict[str, Any]:
        """Get daemon manager statistics."""
        return {
            "running": self._running,
            "total_tasks": len(self._tasks),
            "enabled_tasks": sum(1 for t in self._tasks.values() if t.enabled),
            "tasks": self.list_tasks()
        }


def create_daemon_task(action: Callable, name: str, description: str = ""):
    """Decorator to create a daemon task."""
    task_id = f"daemon_{name.lower().replace(' ', '_')}"
    return {
        "id": task_id,
        "name": name,
        "action": action,
        "description": description
    }


_daemon_manager: Optional[DaemonManager] = None


def get_daemon_manager() -> DaemonManager:
    global _daemon_manager
    if _daemon_manager is None:
        _daemon_manager = DaemonManager()
    return _daemon_manager


def start_daemon_manager():
    manager = get_daemon_manager()
    manager.start()
    return manager


def stop_daemon_manager():
    global _daemon_manager
    if _daemon_manager:
        _daemon_manager.stop()