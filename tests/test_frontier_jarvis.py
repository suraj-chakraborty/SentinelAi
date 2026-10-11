"""
tests/test_frontier_jarvis.py
─────────────────────────────
Comprehensive unit tests for the 5 Frontier JARVIS Capabilities:
1. Nightly Autonomous Multi-Agent Swarm
2. Native Speech-to-Speech Streaming & Acoustic Mirroring
3. Room-Scale Spatial Intelligence & Multi-Room Audio
4. Continuous Video-Rate Screen Understanding
5. Phone Bridge Mobile Hub & Notification Triage
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentinel.core.nightly_swarm import NightlySwarmCoordinator, CodebaseAuditor
from sentinel.commands.swarm_command import SwarmCommand
from sentinel.voice.live_speech_stream import LiveSpeechStream, AdaptiveAcousticMirror
from sentinel.commands.s2s_command import S2SCommand
from sentinel.modules.spatial_iot import SpatialIoTCoordinator
from sentinel.commands.spatial_command import SpatialCommand
from sentinel.vision.continuous_screen_agent import ContinuousScreenAgent
from sentinel.commands.live_vision_command import LiveVisionCommand
from sentinel.core.mobile_hub import MobileHubAgent
from sentinel.commands.mobile_command import MobileCommand
from sentinel.core.intent_detector import IntentDetector, Intent


class TestFrontierJarvisSuite:

    def test_nightly_swarm_and_command(self, tmp_path):
        """Test nightly swarm auditor, tech intelligence, and voice command."""
        coordinator = NightlySwarmCoordinator(workspace_root=str(tmp_path))
        mission = coordinator.run_nightly_mission()
        assert mission["success"] is True
        assert "audit" in mission
        assert os.path.exists(mission["report_path"])

        cmd = SwarmCommand()
        cmd.coordinator = coordinator
        res = cmd.execute("run nightly audit")
        assert "Swarm mission completed" in res

        hunt_res = cmd.execute("hunt tech intelligence")
        assert "intelligence" in hunt_res

    def test_s2s_streaming_and_acoustic_mirror(self):
        """Test adaptive acoustic volume mirroring and live streaming."""
        mirror = AdaptiveAcousticMirror()
        # Empty/synthetic PCM
        normal_analysis = mirror.analyze_audio_energy(b"")
        assert normal_analysis["mode"] == "normal"

        stream = LiveSpeechStream()
        assert stream.connect() is True
        assert stream.is_connected is True

        stream.set_translation_target("ja")
        chunk_analysis = stream.process_incoming_audio_chunk(b"\x00" * 320)
        assert chunk_analysis["target_language"] == "ja"

        cmd = S2SCommand()
        cmd.stream = stream
        s2s_res = cmd.execute("start live speech")
        assert "Sub-200ms latency" in s2s_res

        trans_res = cmd.execute("translate live to spanish")
        assert "SPANISH" in trans_res

    def test_spatial_iot_and_command(self):
        """Test desk lighting sync and multi-room audio hand-off."""
        mock_iot = MagicMock()
        mock_iot.run_scene.return_value = (True, "Scene activated")
        mock_iot.control_home_assistant.return_value = (True, "Entity toggled")

        coordinator = SpatialIoTCoordinator(iot_hub=mock_iot)
        desk_res = coordinator.sync_desk_focus()
        assert desk_res["success"] is True

        cinema_res = coordinator.sync_media_cinema()
        assert cinema_res["success"] is True

        hand_off = coordinator.hand_off_audio_to_room("Kitchen")
        assert hand_off["room"] == "Kitchen"

        cmd = SpatialCommand()
        cmd.coordinator = coordinator
        voice_res = cmd.execute("sync desk lighting")
        assert "Desk context synchronized" in voice_res

    def test_continuous_screen_agent(self):
        """Test continuous screen watcher lifecycle and UI element coordinates."""
        agent = ContinuousScreenAgent(fps=1.0)
        # Should locate element coordinates without crashing
        coords = agent.find_ui_element("Submit")
        assert isinstance(coords, tuple)
        assert len(coords) == 4

        cmd = LiveVisionCommand()
        cmd.agent = agent
        watch_res = cmd.execute("start screen watcher")
        assert "Continuous screen observer active" in watch_res

        where_res = cmd.execute("where is Submit")
        assert "coordinates" in where_res

    def test_mobile_hub_and_triage(self):
        """Test mobile call triage announcement and authenticated remote command."""
        hub = MobileHubAgent(auth_token="secret_token")

        # 1. Incoming call triage
        call_msg = hub.triage_incoming_notification("call", "Sarah", "")
        assert "incoming phone call from Sarah" in call_msg

        # 2. Remote command with invalid token
        reject = hub.execute_remote_command("status", token="wrong_token")
        assert reject["success"] is False

        # 3. Remote command with valid token
        auth_res = hub.execute_remote_command("status", token="secret_token")
        assert auth_res["success"] is True
        assert "Online" in auth_res["message"]

        cmd = MobileCommand()
        cmd.hub = hub
        phone_res = cmd.execute("who called me")
        assert "Sarah" in phone_res

    def test_frontier_intent_detection(self):
        """Test that all 5 frontier features detect correct intents."""
        detector = IntentDetector()

        assert detector.detect_intent("run nightly audit") == Intent.SWARM_MISSION
        assert detector.detect_intent("start live speech") == Intent.S2S_STREAM
        assert detector.detect_intent("sync desk lighting") == Intent.SPATIAL_IOT
        assert detector.detect_intent("start screen watcher") == Intent.CONTINUOUS_VISION
        assert detector.detect_intent("who called me") == Intent.MOBILE_HUB
