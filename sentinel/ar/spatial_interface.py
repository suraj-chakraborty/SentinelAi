"""
sentinel/ar/spatial_interface.py
──────────────────────────────────
Deep Space / Spatial AR Integration Module.

WebXR server for AR hardware integration (Meta Quest/Apple Vision Pro):
- Serve local WebXR server
- Project floating data panels in physical space
- Hand tracking and raycasting
- IoT device spatial mapping
"""

import os
import time
import logging
import threading
import json
import asyncio
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sentinel.app.config import APPDATA_DIR

logger = logging.getLogger("SpatialAR")

AR_DIR = os.path.join(APPDATA_DIR, "ar")
os.makedirs(AR_DIR, exist_ok=True)

WEBXR_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Sentinel AR Interface</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/loaders/GLTFLoader.js"></script>
    <style>
        body { margin: 0; overflow: hidden; }
        #info {
            position: absolute;
            top: 10px;
            width: 100%;
            text-align: center;
            color: white;
            font-family: sans-serif;
            font-size: 18px;
            z-index: 100;
        }
        .panel {
            background: rgba(0, 20, 40, 0.85);
            border: 2px solid #0ea5e9;
            border-radius: 12px;
            padding: 20px;
            color: white;
            font-family: 'Segoe UI', sans-serif;
            box-shadow: 0 0 20px rgba(14, 165, 233, 0.5);
        }
    </style>
</head>
<body>
    <div id="info">Sentinel AR Interface</div>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebXRRenderer({ antialias: true });
        renderer.xr.enabled = true;
        document.body.appendChild(renderer.domElement);
        
        const controller1 = renderer.xr.getController(0);
        const controller2 = renderer.xr.getController(1);
        scene.add(controller1);
        scene.add(controller2);
        
        const geometry = new THREE.BoxGeometry(0.1, 0.1, 0.1);
        const material = new THREE.MeshBasicMaterial({ color: 0x0ea5e9 });
        const cube = new THREE.Mesh(geometry, material);
        cube.position.set(0, 1.5, -1);
        scene.add(cube);
        
        function addPanel(x, y, z, title, content) {
            const panel = document.createElement('div');
            panel.className = 'panel';
            panel.innerHTML = '<h3>' + title + '</h3><p>' + content + '</p>';
            document.body.appendChild(panel);
        }
        
        document.body.addEventListener('click', () => {
            if (navigator.xr) {
                navigator.xr.requestSession('immersive-vr').then((session) => {
                    renderer.xr.setSession(session);
                });
            }
        });
        
        function animate() {
            renderer.setAnimationLoop(render);
        }
        
        function render() {
            renderer.render(scene, camera);
        }
        
        animate();
    </script>
</body>
</html>
"""


@dataclass
class ARPanel:
    """AR panel displayed in physical space."""
    panel_id: str
    title: str
    content: str
    position: tuple
    rotation: tuple
    visible: bool = True


class SpatialMapper:
    """Map physical objects to IoT devices in 3D space."""

    def __init__(self):
        self._spatial_devices: Dict[str, Dict[str, Any]] = {}

    def register_device(
        self,
        device_id: str,
        device_name: str,
        coordinates_3d: tuple,
        device_type: str = "light"
    ):
        """Register a device in physical space."""
        self._spatial_devices[device_id] = {
            "name": device_name,
            "position": coordinates_3d,
            "type": device_type,
            "registered_at": datetime.now().isoformat()
        }

    def find_nearest_device(self, position: tuple, max_distance: float = 1.0) -> Optional[Dict[str, Any]]:
        """Find nearest device to given 3D position."""
        import math
        
        nearest = None
        min_dist = max_distance
        
        for device_id, device in self._spatial_devices.items():
            px, py, pz = position
            dx, dy, dz = device["position"]
            
            distance = math.sqrt((px-dx)**2 + (py-dy)**2 + (pz-dz)**2)
            
            if distance < min_dist:
                min_dist = distance
                nearest = {**device, "id": device_id, "distance": distance}
        
        return nearest

    def get_all_devices(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered spatial devices."""
        return self._spatial_devices


class WebXRServer:
    """WebXR server for AR headset connection."""

    def __init__(self, port: int = 8080):
        self.port = port
        self._running = False
        self._server_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the WebXR server."""
        if self._running:
            return
        
        self._running = True
        self._server_thread = threading.Thread(target=self._run_server, daemon=True, name="WebXRServer")
        self._server_thread.start()
        
        logger.info(f"WebXR server started on port {self.port}")

    def stop(self):
        """Stop the WebXR server."""
        self._running = False
        logger.info("WebXR server stopped")

    def _run_server(self):
        """Run the Flask server for WebXR."""
        try:
            from flask import Flask, send_file, jsonify, request
        except ImportError:
            logger.error("Flask not available for WebXR server")
            return
        
        app = Flask(__name__)
        
        @app.route("/")
        def index():
            return WEBXR_TEMPLATE
        
        @app.route("/api/panels", methods=["GET"])
        def get_panels():
            from sentinel.ar.spatial_interface import get_spatial_interface
            iface = get_spatial_interface()
            return jsonify(iface.get_panels())
        
        @app.route("/api/panels", methods=["POST"])
        def add_panel():
            data = request.json
            from sentinel.ar.spatial_interface import get_spatial_interface
            iface = get_spatial_interface()
            iface.add_panel(data["title"], data["content"], data.get("position", (0, 1.5, -1)))
            return jsonify({"success": True})
        
        @app.route("/api/devices", methods=["GET"])
        def get_devices():
            from sentinel.ar.spatial_interface import get_spatial_interface
            iface = get_spatial_interface()
            return jsonify(iface.get_spatial_devices())
        
        @app.route("/api/devices/register", methods=["POST"])
        def register_device():
            data = request.json
            from sentinel.ar.spatial_interface import get_spatial_interface
            iface = get_spatial_interface()
            iface.register_physical_device(
                data["device_id"],
                data["device_name"],
                data["coordinates"],
                data.get("device_type", "light")
            )
            return jsonify({"success": True})
        
        @app.route("/api/iot/toggle", methods=["POST"])
        def toggle_iot():
            data = request.json
            device_id = data.get("device_id")
            
            try:
                from sentinel.modules.iot_hub import get_iot_hub
                iot = get_iot_hub()
                iot.toggle_device(device_id)
                return jsonify({"success": True, "device": device_id})
            except Exception as e:
                return jsonify({"success": False, "error": str(e)})
        
        try:
            app.run(host="0.0.0.0", port=self.port, debug=False, use_reloader=False)
        except Exception as e:
            logger.error(f"WebXR server error: {e}")


class SpatialInterface:
    """
    Deep Space / Spatial AR Interface.
    
    Features:
    - WebXR server for AR headsets
    - Floating data panels in physical space
    - 3D spatial IoT device mapping
    - Hand tracking integration
    """

    def __init__(
        self,
        webxr_port: int = 8080,
        on_device_toggle: Optional[Callable] = None
    ):
        self.webxr_port = webxr_port
        self.on_device_toggle = on_device_toggle
        
        self._spatial_mapper = SpatialMapper()
        self._webxr_server = WebXRServer(port=webxr_port)
        
        self._panels: Dict[str, ARPanel] = {}
        self._running = False

    def start(self):
        """Start the spatial AR interface."""
        if self._running:
            return
        
        self._running = True
        self._webxr_server.start()
        
        logger.info(f"Spatial AR interface started (WebXR port: {self.webxr_port})")

    def stop(self):
        """Stop the spatial AR interface."""
        self._running = False
        self._webxr_server.stop()
        logger.info("Spatial AR interface stopped")

    def add_panel(
        self,
        title: str,
        content: str,
        position: tuple = (0, 1.5, -1),
        rotation: tuple = (0, 0, 0)
    ) -> str:
        """Add an AR panel to physical space."""
        panel_id = f"panel_{len(self._panels) + 1}"
        
        panel = ARPanel(
            panel_id=panel_id,
            title=title,
            content=content,
            position=position,
            rotation=rotation
        )
        
        self._panels[panel_id] = panel
        
        logger.info(f"Added AR panel: {title} at {position}")
        return panel_id

    def remove_panel(self, panel_id: str) -> bool:
        """Remove an AR panel."""
        if panel_id in self._panels:
            del self._panels[panel_id]
            return True
        return False

    def update_panel(self, panel_id: str, title: Optional[str] = None, content: Optional[str] = None) -> bool:
        """Update an existing panel."""
        if panel_id not in self._panels:
            return False
        
        panel = self._panels[panel_id]
        if title:
            panel.title = title
        if content:
            panel.content = content
        
        return True

    def get_panels(self) -> List[Dict[str, Any]]:
        """Get all AR panels."""
        return [
            {
                "id": p.panel_id,
                "title": p.title,
                "content": p.content,
                "position": p.position,
                "rotation": p.rotation,
                "visible": p.visible
            }
            for p in self._panels.values()
        ]

    def register_physical_device(
        self,
        device_id: str,
        device_name: str,
        coordinates_3d: tuple,
        device_type: str = "light"
    ):
        """Register an IoT device in 3D space."""
        self._spatial_mapper.register_device(device_id, device_name, coordinates_3d, device_type)

    def get_spatial_devices(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered spatial devices."""
        return self._spatial_mapper.get_all_devices()

    def handle_gesture(self, gesture_type: str, position: tuple) -> Dict[str, Any]:
        """Handle AR gesture at given position."""
        if gesture_type == "point":
            device = self._spatial_mapper.find_nearest_device(position)
            
            if device:
                if self.on_device_toggle:
                    self.on_device_toggle(device["id"])
                
                return {
                    "action": "toggle",
                    "device": device["name"],
                    "type": device["type"]
                }
        
        return {"action": "none"}

    def project_system_thermals(self):
        """Project system thermals as floating panel."""
        try:
            import psutil
            
            cpu_temp = psutil.cpu_temperature()
            memory = psutil.virtual_memory()
            
            content = f"""CPU: {cpu_temp}°C
Memory: {memory.percent}%
Processes: {len(psutil.pids())}"""
            
            self.add_panel(
                title="System Thermals",
                content=content,
                position=(0.5, 1.2, -0.5)
            )
        
        except Exception as e:
            logger.error(f"Failed to project thermals: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get spatial interface status."""
        return {
            "running": self._running,
            "webxr_port": self.webxr_port,
            "panels": len(self._panels),
            "devices": len(self._spatial_mapper._spatial_devices)
        }


_spatial_interface: Optional[SpatialInterface] = None


def get_spatial_interface() -> SpatialInterface:
    global _spatial_interface
    if _spatial_interface is None:
        _spatial_interface = SpatialInterface()
    return _spatial_interface


def start_spatial_ar(webxr_port: int = 8080) -> SpatialInterface:
    iface = get_spatial_interface()
    iface.webxr_port = webxr_port
    iface.start()
    return iface


def stop_spatial_ar():
    global _spatial_interface
    if _spatial_interface:
        _spatial_interface.stop()