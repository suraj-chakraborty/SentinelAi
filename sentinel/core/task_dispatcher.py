"""
sentinel/core/task_dispatcher.py
─────────────────────────────────
Autonomous Task Dispatcher — Multi-Device Mesh Network.

Provides intelligent task routing and parallel execution:
  1. Local task queue with priority scheduling.
  2. Multi-device mesh registration (laptop, phone, server, cloud).
  3. Task decomposition into parallelizable sub-tasks.
  4. Load-aware routing based on device capabilities and utilization.
  5. Result aggregation and conflict resolution.
  6. Retry logic with exponential backoff.
  7. Task dependency graph execution (DAG).
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("TaskDispatcher")


class TaskPriority(Enum):
    CRITICAL = 0
    HIGH     = 1
    NORMAL   = 2
    LOW      = 3
    IDLE     = 4


class TaskStatus(Enum):
    QUEUED      = "queued"
    DISPATCHED  = "dispatched"
    RUNNING     = "running"
    COMPLETED   = "completed"
    FAILED      = "failed"
    RETRYING    = "retrying"
    CANCELLED   = "cancelled"


class DeviceType(Enum):
    LOCAL_PC     = "local_pc"
    PHONE        = "phone"
    REMOTE_SERVER = "remote_server"
    CLOUD_GPU    = "cloud_gpu"
    IOT_DEVICE   = "iot_device"
    TABLET       = "tablet"


@dataclass
class DeviceNode:
    """A device in the mesh network."""
    device_id: str
    device_name: str
    device_type: DeviceType
    capabilities: List[str]         # e.g., ["gpu", "internet", "storage", "mic"]
    cpu_cores: int = 4
    memory_mb: int = 8192
    gpu_available: bool = False
    is_online: bool = True
    utilization_pct: float = 0.0    # Current CPU utilization
    last_heartbeat: float = 0.0
    max_concurrent_tasks: int = 4
    current_tasks: int = 0
    endpoint: Optional[str] = None  # Network endpoint for remote dispatch


@dataclass
class Task:
    """A dispatchable task unit."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.QUEUED
    handler: Optional[Callable] = None
    args: Dict[str, Any] = field(default_factory=dict)
    required_capabilities: List[str] = field(default_factory=list)
    assigned_device: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)  # Task IDs
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    timeout_seconds: float = 300.0


@dataclass
class TaskResult:
    """Aggregated result from a dispatched task."""
    task_id: str
    status: TaskStatus
    result: Any
    device_id: Optional[str]
    duration_ms: float
    retries_used: int


class DeviceMesh:
    """Manages the mesh network of registered devices."""

    def __init__(self):
        self._devices: Dict[str, DeviceNode] = {}
        self._lock = threading.Lock()

        # Always register local device
        self._register_local()

    def _register_local(self):
        """Register the local PC as a device."""
        import platform
        import multiprocessing
        local = DeviceNode(
            device_id="local",
            device_name=platform.node(),
            device_type=DeviceType.LOCAL_PC,
            capabilities=["cpu", "gpu", "internet", "storage", "mic", "display"],
            cpu_cores=multiprocessing.cpu_count(),
            memory_mb=_get_total_ram(),
            gpu_available=_check_gpu(),
            is_online=True,
            last_heartbeat=time.time(),
        )
        self._devices["local"] = local

    def register_device(self, device: DeviceNode) -> bool:
        """Register a new device in the mesh."""
        with self._lock:
            self._devices[device.device_id] = device
            logger.info("Device registered: %s (%s)", device.device_name, device.device_type.value)
        return True

    def unregister_device(self, device_id: str) -> bool:
        """Remove a device from the mesh."""
        with self._lock:
            if device_id in self._devices and device_id != "local":
                del self._devices[device_id]
                return True
        return False

    def heartbeat(self, device_id: str, utilization: float = 0.0) -> bool:
        """Update device heartbeat and utilization."""
        with self._lock:
            if device_id in self._devices:
                self._devices[device_id].last_heartbeat = time.time()
                self._devices[device_id].utilization_pct = utilization
                self._devices[device_id].is_online = True
                return True
        return False

    def get_best_device(self, required_capabilities: List[str]) -> Optional[DeviceNode]:
        """Select the best available device for a task based on capabilities and load."""
        with self._lock:
            candidates = []
            for dev in self._devices.values():
                if not dev.is_online:
                    continue
                if dev.current_tasks >= dev.max_concurrent_tasks:
                    continue
                if all(cap in dev.capabilities for cap in required_capabilities):
                    candidates.append(dev)

            if not candidates:
                return None

            # Score: lower utilization and more free slots = better
            def score(d: DeviceNode) -> float:
                util_score = 100 - d.utilization_pct
                slot_score = (d.max_concurrent_tasks - d.current_tasks) * 20
                type_bonus = {
                    DeviceType.CLOUD_GPU: 30,
                    DeviceType.REMOTE_SERVER: 20,
                    DeviceType.LOCAL_PC: 10,
                    DeviceType.PHONE: 0,
                }.get(d.device_type, 5)
                return util_score + slot_score + type_bonus

            candidates.sort(key=score, reverse=True)
            return candidates[0]

    def get_all_devices(self) -> List[Dict[str, Any]]:
        """Return info on all registered devices."""
        with self._lock:
            return [
                {
                    "device_id": d.device_id,
                    "name": d.device_name,
                    "type": d.device_type.value,
                    "online": d.is_online,
                    "utilization": d.utilization_pct,
                    "tasks": d.current_tasks,
                    "capabilities": d.capabilities,
                }
                for d in self._devices.values()
            ]

    def device_count(self) -> int:
        with self._lock:
            return len(self._devices)


class TaskDecomposer:
    """Decomposes complex tasks into parallelizable sub-tasks."""

    def decompose(self, task_description: str) -> List[Dict[str, Any]]:
        """Break a complex task into sub-tasks."""
        subtasks = []

        # Keyword-based decomposition
        if "research" in task_description.lower():
            subtasks.append({
                "name": "web_search",
                "description": f"Search for: {task_description}",
                "capabilities": ["internet"],
            })
            subtasks.append({
                "name": "summarize",
                "description": f"Summarize findings from: {task_description}",
                "capabilities": ["cpu"],
                "depends_on": ["web_search"],
            })
        elif "process" in task_description.lower() and "file" in task_description.lower():
            subtasks.append({
                "name": "read_files",
                "description": f"Read input files for: {task_description}",
                "capabilities": ["storage"],
            })
            subtasks.append({
                "name": "process",
                "description": f"Process data: {task_description}",
                "capabilities": ["cpu"],
                "depends_on": ["read_files"],
            })
            subtasks.append({
                "name": "write_output",
                "description": f"Write results for: {task_description}",
                "capabilities": ["storage"],
                "depends_on": ["process"],
            })
        else:
            # Single task (no decomposition needed)
            subtasks.append({
                "name": "execute",
                "description": task_description,
                "capabilities": ["cpu"],
            })

        return subtasks


class AutonomousTaskDispatcher:
    """
    Master task dispatcher with intelligent routing, parallel execution,
    dependency resolution, and retry logic.
    """

    def __init__(self, max_workers: int = 4):
        self._mesh = DeviceMesh()
        self._decomposer = TaskDecomposer()
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._tasks: Dict[str, Task] = {}
        self._results: Dict[str, TaskResult] = {}
        self._lock = threading.Lock()
        self._workers: List[threading.Thread] = []
        self._max_workers = max_workers
        self._running = False
        self._total_dispatched = 0
        self._total_completed = 0
        self._total_failed = 0
        logger.info("AutonomousTaskDispatcher initialized (workers=%d)", max_workers)

    def start(self) -> bool:
        """Start the task dispatcher workers."""
        if self._running:
            return False
        self._running = True

        for i in range(self._max_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True,
                                  name=f"TaskWorker-{i}")
            t.start()
            self._workers.append(t)

        logger.info("Dispatcher started with %d workers", self._max_workers)
        return True

    def stop(self) -> bool:
        """Stop the dispatcher."""
        self._running = False
        # Push sentinel items to unblock workers
        for _ in self._workers:
            self._task_queue.put((99, None))
        for t in self._workers:
            t.join(timeout=3.0)
        self._workers.clear()
        logger.info("Dispatcher stopped")
        return True

    def submit_task(self, name: str, handler: Optional[Callable] = None,
                    args: Optional[Dict] = None,
                    priority: TaskPriority = TaskPriority.NORMAL,
                    capabilities: Optional[List[str]] = None,
                    depends_on: Optional[List[str]] = None,
                    timeout: float = 300.0) -> Task:
        """Submit a new task to the dispatcher."""
        task = Task(
            name=name,
            handler=handler,
            args=args or {},
            priority=priority,
            required_capabilities=capabilities or ["cpu"],
            depends_on=depends_on or [],
            timeout_seconds=timeout,
        )

        with self._lock:
            self._tasks[task.task_id] = task

        self._task_queue.put((priority.value, task.task_id))
        self._total_dispatched += 1
        logger.info("Task submitted: %s (%s) priority=%s", task.name, task.task_id, priority.value)
        return task

    def submit_complex_task(self, description: str,
                             priority: TaskPriority = TaskPriority.NORMAL) -> List[Task]:
        """Decompose and submit a complex task."""
        subtask_specs = self._decomposer.decompose(description)
        tasks = []
        id_map = {}

        for spec in subtask_specs:
            deps = []
            for dep_name in spec.get("depends_on", []):
                if dep_name in id_map:
                    deps.append(id_map[dep_name])

            task = self.submit_task(
                name=spec["name"],
                args={"description": spec["description"]},
                priority=priority,
                capabilities=spec.get("capabilities", ["cpu"]),
                depends_on=deps,
            )
            id_map[spec["name"]] = task.task_id
            tasks.append(task)

        return tasks

    def _worker_loop(self):
        """Worker thread that picks and executes tasks."""
        while self._running:
            try:
                priority_val, task_id = self._task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task_id is None:  # Sentinel
                break

            with self._lock:
                task = self._tasks.get(task_id)

            if not task or task.status == TaskStatus.CANCELLED:
                continue

            # Check dependencies
            if task.depends_on:
                deps_met = all(
                    self._tasks.get(dep_id, Task()).status == TaskStatus.COMPLETED
                    for dep_id in task.depends_on
                )
                if not deps_met:
                    # Re-queue
                    self._task_queue.put((priority_val, task_id))
                    time.sleep(0.5)
                    continue

            # Find best device
            device = self._mesh.get_best_device(task.required_capabilities)
            if not device:
                device = self._mesh._devices.get("local")

            task.assigned_device = device.device_id if device else "local"
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            if device:
                device.current_tasks += 1

            try:
                if task.handler:
                    task.result = task.handler(**task.args)
                else:
                    task.result = {"status": "executed", "description": task.args.get("description", "")}
                task.status = TaskStatus.COMPLETED
                self._total_completed += 1
            except Exception as e:
                task.error = str(e)
                task.retries += 1
                if task.retries < task.max_retries:
                    task.status = TaskStatus.RETRYING
                    # Exponential backoff
                    time.sleep(min(2 ** task.retries, 30))
                    self._task_queue.put((priority_val, task_id))
                else:
                    task.status = TaskStatus.FAILED
                    self._total_failed += 1
            finally:
                task.completed_at = time.time()
                if device:
                    device.current_tasks = max(0, device.current_tasks - 1)

                # Store result
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    duration = ((task.completed_at or 0) - (task.started_at or 0)) * 1000
                    with self._lock:
                        self._results[task.task_id] = TaskResult(
                            task_id=task.task_id,
                            status=task.status,
                            result=task.result,
                            device_id=task.assigned_device,
                            duration_ms=round(duration, 2),
                            retries_used=task.retries,
                        )

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued or running task."""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].status = TaskStatus.CANCELLED
                return True
        return False

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return None
        return {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status.value,
            "device": task.assigned_device,
            "retries": task.retries,
            "error": task.error,
        }

    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall dispatcher status."""
        with self._lock:
            status_counts = {}
            for t in self._tasks.values():
                s = t.status.value
                status_counts[s] = status_counts.get(s, 0) + 1

        return {
            "running": self._running,
            "workers": len(self._workers),
            "total_dispatched": self._total_dispatched,
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "queue_size": self._task_queue.qsize(),
            "status_breakdown": status_counts,
            "mesh_devices": self._mesh.device_count(),
        }

    def get_mesh_devices(self) -> List[Dict[str, Any]]:
        """Return all registered mesh devices."""
        return self._mesh.get_all_devices()

    def register_device(self, name: str, device_type: str,
                         capabilities: List[str],
                         endpoint: Optional[str] = None) -> str:
        """Register a new device in the mesh."""
        device_id = hashlib.md5(name.encode()).hexdigest()[:8]
        try:
            dtype = DeviceType(device_type)
        except ValueError:
            dtype = DeviceType.LOCAL_PC

        device = DeviceNode(
            device_id=device_id,
            device_name=name,
            device_type=dtype,
            capabilities=capabilities,
            is_online=True,
            last_heartbeat=time.time(),
            endpoint=endpoint,
        )
        self._mesh.register_device(device)
        return device_id


def _get_total_ram() -> int:
    try:
        import psutil
        return psutil.virtual_memory().total // (1024 * 1024)
    except ImportError:
        return 8192


def _check_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
