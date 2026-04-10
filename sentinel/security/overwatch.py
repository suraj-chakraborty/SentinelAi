"""
sentinel/security/overwatch.py
──────────────────────────────
Real-Time Kernel-Level Cybersecurity Module.

AI-driven process and network monitor that actively hunts threats:
- Monitor outbound network packets
- Track newly spawned processes
- Suspend suspicious threads
- Quarantine and generate reports
"""

import os
import time
import logging
import threading
import queue
import json
import hashlib
import subprocess
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import psutil

from sentinel.app.config import APPDATA_DIR

logger = logging.getLogger("CyberOverwatch")

OVERWATCH_DIR = os.path.join(APPDATA_DIR, "security", "overwatch")
os.makedirs(OVERWATCH_DIR, exist_ok=True)

QUARANTINE_DIR = os.path.join(OVERWATCH_DIR, "quarantine")
os.makedirs(QUARANTINE_DIR, exist_ok=True)

REPORTS_DIR = os.path.join(OVERWATCH_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


@dataclass
class ThreatEvent:
    """Detected threat event."""
    id: str
    timestamp: datetime
    threat_type: str
    severity: str
    process_name: str
    process_pid: int
    description: str
    source_ip: Optional[str] = None
    dest_ip: Optional[str] = None
    action_taken: Optional[str] = None
    user_response: Optional[str] = None


class ProcessMonitor:
    """Monitor process creation and behavior."""

    def __init__(self):
        self._known_processes: Dict[int, dict] = {}
        self._suspicious_patterns = [
            "payload", "inject", "shellcode", "exploit", "keylog",
            "cryptominer", "miner", "reverse_shell", "backdoor"
        ]
        self._monitored_pids: set = set()

    def get_all_processes(self) -> List[Dict[str, Any]]:
        """Get all running processes."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'exe', 'cmdline', 'create_time']):
            try:
                info = proc.info
                processes.append({
                    'pid': info.get('pid'),
                    'name': info.get('name'),
                    'exe': info.get('exe'),
                    'cmdline': info.get('cmdline'),
                    'create_time': info.get('create_time')
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return processes

    def detect_new_processes(self) -> List[Dict[str, Any]]:
        """Detect newly spawned processes."""
        current_pids = {p['pid'] for p in self.get_all_processes()}
        new_pids = current_pids - self._monitored_pids
        
        new_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'exe', 'cmdline', 'create_time']):
            try:
                if proc.info['pid'] in new_pids:
                    new_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'exe': proc.info['exe'],
                        'cmdline': proc.info['cmdline'],
                        'create_time': proc.info['create_time'],
                        'parent_pid': self._get_parent_pid(proc)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        self._monitored_pids = current_pids
        return new_processes

    def _get_parent_pid(self, proc) -> Optional[int]:
        try:
            return proc.parent().pid if proc.parent() else None
        except Exception:
            return None

    def is_suspicious(self, proc_info: Dict[str, Any]) -> bool:
        """Check if process matches suspicious patterns."""
        name = proc_info.get('name', '').lower()
        exe = proc_info.get('exe', '').lower()
        cmdline = ' '.join(proc_info.get('cmdline', [])).lower()
        
        for pattern in self._suspicious_patterns:
            if pattern in name or pattern in exe or pattern in cmdline:
                return True
        
        return False


class NetworkMonitor:
    """Monitor network connections for suspicious activity."""

    def __init__(self):
        self._trusted_ips: set = {
            '127.0.0.1', 'localhost', '::1',
            '192.168.0.0/16', '10.0.0.0/8', '172.16.0.0/12'
        }
        self._suspicious_ports = {4444, 5555, 6666, 31337, 1337}
        self._active_connections: Dict[int, List[dict]] = {}

    def get_active_connections(self) -> Dict[int, List[dict]]:
        """Get all active network connections."""
        connections = {}
        
        for conn in psutil.net_connections(kind='inet'):
            try:
                if conn.status == 'ESTABLISHED' and conn.raddr:
                    pid = conn.pid
                    if pid not in connections:
                        connections[pid] = []
                    
                    connections[pid].append({
                        'local_addr': conn.laddr.ip if conn.laddr else None,
                        'local_port': conn.laddr.port if conn.laddr else None,
                        'remote_addr': conn.raddr.ip,
                        'remote_port': conn.raddr.port,
                        'status': conn.status
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return connections

    def is_suspicious_connection(self, conn: dict) -> bool:
        """Check if connection is suspicious."""
        remote_ip = conn.get('remote_addr', '')
        remote_port = conn.get('remote_port', 0)
        
        if self._is_private_ip(remote_ip):
            return False
        
        if remote_port in self._suspicious_ports:
            return True
        
        return False

    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is private."""
        try:
            import ipaddress
            return ipaddress.ip_address(ip).is_private
        except Exception:
            return False


class CyberOverwatch:
    """
    Real-time cybersecurity monitor and threat detector.
    
    Features:
    - Process monitoring withSuspension capability
    - Network connection tracking
    - Threat detection and alerting
    - Quarantine and reporting
    """

    def __init__(
        self,
        on_threat_detected: Optional[Callable] = None,
        on_user_action: Optional[Callable] = None
    ):
        self.on_threat_detected = on_threat_detected
        self.on_user_action = on_user_action
        
        self._process_monitor = ProcessMonitor()
        self._network_monitor = NetworkMonitor()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self._threat_queue: queue.Queue = queue.Queue()
        self._threat_history: List[ThreatEvent] = []
        
        self._suspended_pids: set = set
        
        self._scan_interval_sec = 2
        self._max_history = 100

    def start(self):
        """Start the overwatch daemon."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="CyberOverwatch")
        self._thread.start()
        
        logger.info("Cyber Overwatch started")

    def stop(self):
        """Stop the overwatch daemon."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Cyber Overwatch stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._check_processes()
                self._check_network()
                time.sleep(self._scan_interval_sec)
            except Exception as e:
                logger.error(f"Overwatch error: {e}")
                time.sleep(self._scan_interval_sec)

    def _check_processes(self):
        """Check for new or suspicious processes."""
        new_procs = self._process_monitor.detect_new_processes()
        
        for proc in new_procs:
            if self._process_monitor.is_suspicious(proc):
                self._handle_threat(
                    threat_type="suspicious_process",
                    severity="high",
                    process_name=proc.get('name', 'unknown'),
                    process_pid=proc.get('pid'),
                    description=f"Suspicious process detected: {proc.get('name')}",
                    cmdline=proc.get('cmdline', [])
                )
            else:
                logger.debug(f"New process: {proc.get('name')} (PID: {proc.get('pid')})")

    def _check_network(self):
        """Check for suspicious network connections."""
        connections = self._network_monitor.get_active_connections()
        
        for pid, conns in connections.items():
            try:
                proc = psutil.Process(pid)
                proc_name = proc.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                proc_name = f"PID {pid}"
            
            for conn in conns:
                if self._network_monitor.is_suspicious_connection(conn):
                    self._handle_threat(
                        threat_type="suspicious_connection",
                        severity="medium",
                        process_name=proc_name,
                        process_pid=pid,
                        description=f"Suspicious outbound connection to {conn.get('remote_addr')}:{conn.get('remote_port')}",
                        dest_ip=conn.get('remote_addr')
                    )

    def _handle_threat(
        self,
        threat_type: str,
        severity: str,
        process_name: str,
        process_pid: int,
        description: str,
        **kwargs
    ):
        """Handle detected threat."""
        threat_event = ThreatEvent(
            id=str(hashlib.md5(f"{process_pid}{datetime.now()}".encode()).hexdigest())[:12],
            timestamp=datetime.now(),
            threat_type=threat_type,
            severity=severity,
            process_name=process_name,
            process_pid=process_pid,
            description=description,
            **kwargs
        )
        
        self._threat_queue.put(threat_event)
        self._threat_history.append(threat_event)
        
        if len(self._threat_history) > self._max_history:
            self._threat_history = self._threat_history[-self._max_history:]
        
        logger.warning(f"Threat detected: {threat_type} - {process_name} (PID: {process_pid})")
        
        if self.on_threat_detected:
            self.on_threat_detected(threat_event)

    def suspend_process(self, pid: int) -> bool:
        """Suspend a suspicious process."""
        try:
            proc = psutil.Process(pid)
            proc.suspend()
            self._suspended_pids.add(pid)
            
            logger.info(f"Suspended process: {pid}")
            return True
        except Exception as e:
            logger.error(f"Failed to suspend process {pid}: {e}")
            return False

    def resume_process(self, pid: int) -> bool:
        """Resume a suspended process."""
        try:
            proc = psutil.Process(pid)
            proc.resume()
            self._suspended_pids.discard(pid)
            
            logger.info(f"Resumed process: {pid}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume process {pid}: {e}")
            return False

    def terminate_process(self, pid: int) -> bool:
        """Terminate a malicious process."""
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            proc.wait(timeout=5)
            
            self._suspended_pids.discard(pid)
            logger.info(f"Terminated process: {pid}")
            return True
        except Exception as e:
            logger.error(f"Failed to terminate process {pid}: {e}")
            return False

    def quarantine_process(self, pid: int) -> bool:
        """Quarantine a malicious process and its files."""
        try:
            proc = psutil.Process(pid)
            exe_path = proc.exe()
            
            if exe_path and os.path.exists(exe_path):
                exe_name = os.path.basename(exe_path)
                quarantine_path = os.path.join(QUARANTINE_DIR, f"{exe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                import shutil
                shutil.move(exe_path, quarantine_path)
                
                logger.info(f"Quarantined: {exe_path} -> {quarantine_path}")
            
            self.terminate_process(pid)
            return True
        except Exception as e:
            logger.error(f"Failed to quarantine process {pid}: {e}")
            return False

    def generate_report(self, threat_event: ThreatEvent) -> str:
        """Generate post-incident cybersecurity report."""
        report = {
            "incident_id": threat_event.id,
            "timestamp": threat_event.timestamp.isoformat(),
            "threat_type": threat_event.threat_type,
            "severity": threat_event.severity,
            "process": {
                "name": threat_event.process_name,
                "pid": threat_event.process_pid
            },
            "description": threat_event.description,
            "action_taken": threat_event.action_taken,
            "user_response": threat_event.user_response,
            "remediation": "Process terminated and quarantined" if threat_event.action_taken == "terminated" else "None"
        }
        
        filepath = os.path.join(
            REPORTS_DIR,
            f"incident_{threat_event.id}_{threat_event.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Incident report saved: {filepath}")
        return filepath

    def get_pending_threats(self) -> List[ThreatEvent]:
        """Get pending threat events."""
        threats = []
        while True:
            try:
                threat = self._threat_queue.get_nowait()
                threats.append(threat)
            except queue.Empty:
                break
        return threats

    def get_threat_history(self) -> List[Dict[str, Any]]:
        """Get threat history."""
        return [
            {
                "id": t.id,
                "timestamp": t.timestamp.isoformat(),
                "threat_type": t.threat_type,
                "severity": t.severity,
                "process_name": t.process_name,
                "process_pid": t.process_pid,
                "description": t.description,
                "action_taken": t.action_taken
            }
            for t in self._threat_history
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get overwatch statistics."""
        return {
            "running": self._running,
            "threats_detected": len(self._threat_history),
            "suspended_processes": len(self._suspended_pids),
            "pending_threats": self._threat_queue.qsize()
        }


_overwatch: Optional[CyberOverwatch] = None


def get_overwatch() -> CyberOverwatch:
    global _overwatch
    if _overwatch is None:
        _overwatch = CyberOverwatch()
    return _overwatch


def start_overwatch() -> CyberOverwatch:
    overwatch = get_overwatch()
    overwatch.start()
    return overwatch


def stop_overwatch():
    global _overwatch
    if _overwatch:
        overwatch.stop()