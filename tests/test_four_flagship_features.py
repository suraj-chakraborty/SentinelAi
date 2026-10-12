"""
tests/test_four_flagship_features.py
───────────────────────────────────
Unit tests for the 4 flagship functionalities:
1. Real-Time 3D Viseme Lip-Sync (Audio-reactive avatar morphing).
2. Live Screen Vision HUD (Proactive error detection & bounding boxes).
3. Browser-Native Hands-Free Voice Streaming (Sub-300ms S2S over WebSocket).
4. Transparent Floating Desktop Widget (Win + Space companion & hotkey hooks).
"""

import importlib.util
import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

_PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None


class TestFourFlagshipFeatures(unittest.TestCase):
    def setUp(self):
        from sentinel.core.web_server import SentinelWebServer

        self.orchestrator = MagicMock()
        self.orchestrator.run_command.return_value = "Mocked execution successful"
        self.server = SentinelWebServer(orchestrator=self.orchestrator)
        self.client = TestClient(self.server.app)

    # ── Feature 1: Real-Time 3D Viseme Lip-Sync ──────────────────────────────
    def test_viseme_hologram_in_index_html(self):
        tmpl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sentinel", "templates", "index.html")
        self.assertTrue(os.path.exists(tmpl_path), "index.html must exist")
        with open(tmpl_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Check Three.js import and container
        self.assertIn("three.min.js", html)
        self.assertIn("three-canvas-container", html)

        # Check formant decomposition & viseme morph logic
        self.assertIn("animateHologram", html)
        self.assertIn("VISEME: OH / AA · OPEN", html)
        self.assertIn("VISEME: EE · WIDE", html)
        self.assertIn("VISEME: TH / SS · SIBILANT", html)
        self.assertIn("innerCore.scale.set", html)

    # ── Feature 2: Live Screen Vision HUD ────────────────────────────────────
    @unittest.skipUnless(_PIL_AVAILABLE, "PIL/Pillow not installed")
    @patch("sentinel.utils.window_context.get_active_window_info")
    @patch("PIL.ImageGrab.grab")
    def test_screen_vision_hud_endpoint(self, mock_grab, mock_window):
        from PIL import Image

        mock_img = Image.new("RGB", (640, 360), color="blue")
        mock_grab.return_value = mock_img
        mock_window.return_value = {
            "title": "Exception in thread 'main' - Visual Studio Code",
            "app_name": "Code",
            "pid": 1234,
        }

        resp = self.client.get("/vision/screen/inspect")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertTrue(data.get("success"))
        self.assertIn("frame", data)
        self.assertTrue(len(data.get("bounding_boxes", [])) > 0)
        # Should detect 'exception' in window title and flag an error box
        boxes = data.get("bounding_boxes", [])
        self.assertEqual(boxes[0]["type"], "error")
        self.assertIn("alerts", data)
        self.assertTrue(len(data["alerts"]) > 0)

    # ── Feature 3: Browser-Native Hands-Free Voice Streaming ─────────────────
    def test_hands_free_voice_in_index_html(self):
        tmpl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sentinel", "templates", "index.html")
        with open(tmpl_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Check VAD & streaming elements
        self.assertIn("toggleHandsFreeVoice", html)
        self.assertIn("setupAudioVAD", html)
        self.assertIn("startCapturingUtterance", html)
        self.assertIn("finishCapturingUtterance", html)
        self.assertIn("voice_audio", html)
        self.assertIn("triggerBargeIn", html)

    def test_websocket_barge_in_and_command(self):
        with self.client.websocket_connect(f"/ws?key={self.server._api_key}") as ws:
            # Expect initial connection confirmation
            welcome = ws.receive_json()
            self.assertEqual(welcome.get("type"), "connected")

            # Test barge-in trigger over WebSocket
            ws.send_json({"type": "barge_in"})
            barge_resp = ws.receive_json()
            self.assertEqual(barge_resp.get("type"), "barge_in")

            # Test text command execution over WebSocket
            ws.send_json({"type": "command", "text": "status check"})
            resp = ws.receive_json()
            self.assertEqual(resp.get("type"), "response")
            self.assertEqual(resp.get("command"), "status check")

    # ── Feature 4: Transparent Floating Desktop Widget ───────────────────────
    def test_floating_widget_companion_classes(self):
        from sentinel.app.floating_widget import TransparentFloatingWidget
        from sentinel.commands.companion_command import CompanionCommand
        from sentinel.utils.hotkey_manager import GlobalHotkeyManager

        self.assertIsNotNone(TransparentFloatingWidget)
        self.assertIsNotNone(GlobalHotkeyManager)
        cmd = CompanionCommand(self.orchestrator)
        result = cmd.execute("toggle companion")
        self.assertIn("companion", result.lower())

    def test_companion_status_endpoint(self):
        resp = self.client.get("/companion/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("shortcut", data)
        self.assertIn("Win + Space", data["shortcut"])

    # ── Security Shield Process Normalization ────────────────────────────────
    @patch("psutil.process_iter")
    def test_security_shield_terminal_alert_normalization(self, mock_iter):
        from sentinel.modules.security_shield import SecurityShieldModule

        notifier = MagicMock()
        shield = SecurityShieldModule(notifier)

        # 1. Owner's PowerShell e.g. "DESKTOP-ABC\suraj"
        owner_proc = MagicMock()
        owner_proc.info = {
            "pid": 1111,
            "name": "powershell.exe",
            "username": f"DESKTOP-PC\\{os.environ.get('USERNAME', 'suraj')}",
        }

        # 2. Windows System cmd.exe
        sys_proc = MagicMock()
        sys_proc.info = {"pid": 2222, "name": "cmd.exe", "username": "NT AUTHORITY\\SYSTEM"}

        # 3. Foreign rogue process
        rogue_proc = MagicMock()
        rogue_proc.info = {"pid": 9999, "name": "powershell.exe", "username": "UNKNOWN_DOMAIN\\intruder"}

        mock_iter.return_value = [owner_proc, sys_proc, rogue_proc]

        shield.is_running = True
        # Run one pass of process check logic directly:
        with patch("time.sleep", side_effect=lambda s: setattr(shield, "is_running", False)):
            shield._shield_loop()

        # Should only alert on the rogue process, not the owner or system!
        self.assertEqual(notifier.show_notification.call_count, 1)
        alert_args = notifier.show_notification.call_args[0]
        self.assertIn("9999", alert_args[1])


if __name__ == "__main__":
    unittest.main()
