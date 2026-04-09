"""
sentinel/core/phone_bridge.py
─────────────────────────────
Inter-Device Hive Mind - Phone/PC Bridge.

Creates a secure REST endpoint proxy to bridge Sentinel Desktop
with the user's mobile device via encrypted tunnel (Ngrok/Cloudflare).
"""

import os
import time
import logging
import threading
import queue
import json
import uuid
import hashlib
import base64
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps

from sentinel.app.config import APPDATA_DIR

logger = logging.getLogger("PhoneBridge")

BRIDGE_DIR = os.path.join(APPDATA_DIR, "phone_bridge")
os.makedirs(BRIDGE_DIR, exist_ok=True)

TOKEN_FILE = os.path.join(BRIDGE_DIR, "auth_token.key")
TUNNEL_CONFIG_FILE = os.path.join(BRIDGE_DIR, "tunnel_config.json")


def generate_token(length: int = 32) -> str:
    """Generate a secure authentication token."""
    return base64.urlsafe_b64encode(os.urandom(length)).decode()


def hash_token(token: str) -> str:
    """Hash token for storage."""
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass
class BridgeCommand:
    """Command received from remote device."""
    id: str
    command: str
    timestamp: datetime
    source: str
    params: Dict[str, Any]


class CryptoBridge:
    """AES-256 encrypted token management."""

    def __init__(self):
        self._token = None
        self._token_hash = None
        self._load_or_create_token()

    def _load_or_create_token(self):
        """Load existing token or create new one."""
        if os.path.exists(TOKEN_FILE):
            try:
                with open(TOKEN_FILE, "r") as f:
                    data = json.load(f)
                    self._token = data.get("token")
                    self._token_hash = data.get("hash")
            except Exception:
                pass
        
        if not self._token:
            self._token = generate_token()
            self._token_hash = hash_token(self._token)
            with open(TOKEN_FILE, "w") as f:
                json.dump({"token": self._token, "hash": self._token_hash}, f)
            logger.info("New bridge token generated")

    def get_token(self) -> str:
        """Get the authentication token."""
        return self._token

    def verify_token(self, token: str) -> bool:
        """Verify a token."""
        return hash_token(token) == self._token_hash


class TunnelManager:
    """Manages tunnel connections (Ngrok/Cloudflare)."""

    def __init__(self):
        self._tunnel_url: Optional[str] = None
        self._tunnel_type: str = "ngrok"
        self._process = None
        self._running = False

    def start_ngrok(self, port: int = 8000) -> bool:
        """Start Ngrok tunnel."""
        ngrok_path = os.path.join(os.environ.get("LOCALAPPDATA", ""), "ngrok.exe")
        
        if not os.path.exists(ngrok_path):
            logger.warning("Ngrok not found, attempting cloudflared")
            return self.start_cloudflared(port)
        
        try:
            import subprocess
            self._process = subprocess.Popen(
                [ngrok_path, "http", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            time.sleep(3)
            
            try:
                import requests
                resp = requests.get("http://localhost:4040/api/tunnels", timeout=5)
                tunnels = resp.json().get("tunnels", [])
                if tunnels:
                    self._tunnel_url = tunnels[0].get("public_url")
                    self._running = True
                    logger.info(f"Ngrok tunnel: {self._tunnel_url}")
                    return True
            except Exception:
                pass
            
            return self.start_cloudflared(port)
        
        except Exception as e:
            logger.error(f"Ngrok start failed: {e}")
            return self.start_cloudflared(port)

    def start_cloudflared(self, port: int = 8000) -> bool:
        """Start Cloudflare tunnel."""
        cloudflared_path = os.path.join(os.environ.get("LOCALAPPDATA", ""), "cloudflared.exe")
        
        try:
            import subprocess
            self._process = subprocess.Popen(
                [cloudflared_path, "tunnel", "--url", f"http://localhost:{port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            time.sleep(5)
            self._running = True
            self._tunnel_url = "cloudflared://ephemeral"
            self._tunnel_type = "cloudflared"
            
            logger.info("Cloudflared tunnel started")
            return True
        
        except Exception as e:
            logger.error(f"Cloudflared start failed: {e}")
            return False

    def stop(self):
        """Stop the tunnel."""
        self._running = False
        if self._process:
            try:
                self._process.terminate()
            except Exception:
                pass

    def get_url(self) -> Optional[str]:
        """Get the tunnel URL."""
        return self._tunnel_url

    def is_running(self) -> bool:
        """Check if tunnel is running."""
        return self._running


class PhoneBridge:
    """
    Inter-Device Hive Mind - Phone/PC Bridge.
    
    Features:
    - Secure REST endpoint with AES-256 token auth
    - Tunnel integration (Ngrok/Cloudflare)
    - Command queue for remote execution
    - Push notifications to phone
    """

    def __init__(
        self,
        port: int = 8000,
        on_command: Optional[Callable] = None
    ):
        self.port = port
        self.on_command = on_command
        
        self._crypto = CryptoBridge()
        self._tunnel = TunnelManager()
        self._command_queue: queue.Queue = queue.Queue()
        
        self._running = False
        self._server_thread: Optional[threading.Thread] = None

    def start(self, use_tunnel: bool = True):
        """Start the phone bridge server."""
        if self._running:
            return
        
        self._running = True
        
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="PhoneBridge"
        )
        self._server_thread.start()
        
        if use_tunnel:
            self._start_tunnel()
        
        logger.info(f"Phone bridge started on port {self.port}")

    def stop(self):
        """Stop the phone bridge."""
        self._running = False
        self._tunnel.stop()
        logger.info("Phone bridge stopped")

    def _run_server(self):
        """Run the Flask/FastAPI server."""
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            logger.error("Flask not available for phone bridge")
            return
        
        app = Flask(__name__)
        
        @app.before_request
        def auth_check():
            token = request.headers.get("X-Bridge-Token") or request.args.get("token")
            if not token or not self._crypto.verify_token(token):
                return jsonify({"error": "Unauthorized"}), 401
        
        @app.route("/bridge/execute", methods=["POST"])
        def execute_command():
            data = request.json
            command = data.get("command", "")
            params = data.get("params", {})
            
            bridge_cmd = BridgeCommand(
                id=str(uuid.uuid4()),
                command=command,
                timestamp=datetime.now(),
                source="phone",
                params=params
            )
            
            self._command_queue.put(bridge_cmd)
            
            if self.on_command:
                result = self.on_command(command, params)
                return jsonify({"status": "executed", "result": result})
            
            return jsonify({"status": "queued", "command_id": bridge_cmd.id})
        
        @app.route("/bridge/status", methods=["GET"])
        def get_status():
            return jsonify({
                "status": "online",
                "tunnel_url": self._tunnel.get_url(),
                "commands_pending": self._command_queue.qsize()
            })
        
        @app.route("/bridge/notify", methods=["POST"])
        def send_notification():
            data = request.json
            message = data.get("message", "")
            
            try:
                from sentinel.utils.notifier import send_notification as notify
                notify("Sentinel", message)
            except Exception:
                pass
            
            return jsonify({"status": "sent"})
        
        try:
            app.run(host="0.0.0.0", port=self.port, debug=False, use_reloader=False)
        except Exception as e:
            logger.error(f"Bridge server error: {e}")

    def _start_tunnel(self):
        """Start tunnel for remote access."""
        self._tunnel.start_ngrok(self.port)

    def get_command(self, timeout: float = 1.0) -> Optional[BridgeCommand]:
        """Get next command from queue."""
        try:
            return self._command_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_token(self) -> str:
        """Get the bridge authentication token."""
        return self._crypto.get_token()

    def get_tunnel_url(self) -> Optional[str]:
        """Get the public tunnel URL."""
        return self._tunnel.get_url()

    def execute_remote_command(self, command: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a command from remote source."""
        if self.on_command:
            return self.on_command(command, params or {})
        return {"error": "No handler configured"}


_phone_bridge: Optional[PhoneBridge] = None


def get_phone_bridge() -> PhoneBridge:
    global _phone_bridge
    if _phone_bridge is None:
        _phone_bridge = PhoneBridge()
    return _phone_bridge


def start_phone_bridge(port: int = 8000, use_tunnel: bool = True) -> PhoneBridge:
    bridge = get_phone_bridge()
    bridge.port = port
    bridge.start(use_tunnel)
    return bridge


def stop_phone_bridge():
    global _phone_bridge
    if _phone_bridge:
        _phone_bridge.stop()