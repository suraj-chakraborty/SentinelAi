"""
tests/test_frontier_jarvis_v2.py
─────────────────────────────────
Comprehensive test suite for the 7 Next-Generation Frontier JARVIS capabilities:
1. Holographic 3D Floating Avatar & Real-Time Lip-Sync
2. Webcam Spatial Air-Gestures & Touchless OS Control
3. Cognitive "Dream State" & Autonomous Memory Consolidation
4. NPU / DirectML Neural Hardware Acceleration
5. Contactless Biometric Vitals & Circadian Health Guardian
6. Autonomous Voice Pair-Programmer (Code Whisperer)
7. Autonomous Task Dispatcher & Multi-Device Mesh
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentinel.ar.holographic_avatar import AvatarEmotion, HolographicAvatarEngine, Viseme
from sentinel.commands.avatar_command import AvatarCommand
from sentinel.commands.code_whisperer_command import CodeWhispererCommand
from sentinel.commands.dispatcher_command import DispatcherCommand
from sentinel.commands.dream_command import DreamCommand
from sentinel.commands.gesture_command import GestureCommand
from sentinel.commands.npu_command import NPUCommand
from sentinel.commands.vitals_command import VitalsCommand
from sentinel.core.command_router import CommandRouter
from sentinel.core.intent_detector import Intent, IntentDetector
from sentinel.core.task_dispatcher import AutonomousTaskDispatcher, TaskPriority
from sentinel.daemons.dream_state import DreamPhase, DreamStateEngine
from sentinel.modules.code_whisperer import CodeLanguage, CodeWhispererEngine
from sentinel.neuro.npu_accelerator import (
    NeuralAccelerationEngine,
    TaskType,
)
from sentinel.perception.air_gesture import AirGestureController
from sentinel.perception.biometric_vitals import BiometricVitalsEngine


class TestFrontierJarvisV2Suite:
    # ── Feature 1: Holographic 3D Floating Avatar ────────────────────────────
    def test_holographic_avatar_engine_and_command(self):
        """Test avatar emotions, lip-sync feed, and voice command plugin."""
        avatar = HolographicAvatarEngine()
        trans = avatar.set_emotion(AvatarEmotion.CUTE_ANGER)
        assert trans["current"] == AvatarEmotion.CUTE_ANGER.value
        assert avatar.get_emotion() == AvatarEmotion.CUTE_ANGER

        # Test lip-sync feed
        viseme_data = avatar.feed_audio(b"\x00\x50\x10\x40" * 100)
        assert "viseme" in viseme_data
        assert isinstance(viseme_data["viseme"], Viseme)

        # Test AvatarCommand plugin
        cmd = AvatarCommand()
        cmd.avatar = avatar

        show_res = cmd.execute("show avatar")
        assert "activated" in show_res.lower() or "visible" in show_res.lower()

        pout_res = cmd.execute("avatar cute anger")
        assert "cute anger" in pout_res.lower() or "pout" in pout_res.lower()

        flirt_res = cmd.execute("avatar flirt")
        assert "flirty" in flirt_res.lower() or "flirt" in flirt_res.lower()

    # ── Feature 2: Air Gestures ──────────────────────────────────────────────
    def test_air_gesture_controller_and_command(self):
        """Test gesture status, controller lifecycle, and command plugin."""
        controller = AirGestureController()
        status = controller.get_status()
        assert "running" in status
        assert "total_gestures_detected" in status

        cmd = GestureCommand()
        cmd.controller = controller

        status_res = cmd.execute("gesture status")
        assert "gesture control" in status_res.lower()

        stop_res = cmd.execute("stop gesture control")
        assert "deactivated" in stop_res.lower()

    # ── Feature 3: Dream State & Memory Consolidation ────────────────────────
    def test_dream_state_engine_and_command(self, tmp_path):
        """Test dream state autonomous maintenance, reflection, and morning briefing."""
        engine = DreamStateEngine(workspace_roots=[str(tmp_path)])
        enter_res = engine.enter_dream()
        assert enter_res["success"] is True
        status = engine.get_status()
        assert "phase" in status

        wake_res = engine.wake_up()
        assert wake_res["success"] is True
        assert wake_res["phase"] == DreamPhase.AWAKE.value

        cmd = DreamCommand()
        cmd.engine = engine

        status_res = cmd.execute("dream status")
        assert "dream state" in status_res.lower()

        report_res = cmd.execute("morning report")
        assert "report" in report_res.lower() and "completed" in report_res.lower()

    # ── Feature 4: NPU / Neural Hardware Acceleration ────────────────────────
    def test_npu_accelerator_and_command(self):
        """Test hardware detection, quantization advisory, routing, and command."""
        engine = NeuralAccelerationEngine()
        init_res = engine.initialize()

        assert "backends_available" in init_res
        assert "primary" in init_res

        # Quantization advice
        advice = engine.recommend_quantization(model_size_mb=4000.0, task_type=TaskType.LLM_INFERENCE)
        assert "recommended" in advice
        assert advice["recommended"] in ("fp16", "int8", "int4")

        cmd = NPUCommand()
        cmd.engine = engine

        status_res = cmd.execute("npu status")
        assert "neural accelerator" in status_res.lower()

        temp_res = cmd.execute("gpu temperature")
        assert "gpu stats" in temp_res.lower()

    # ── Feature 5: Biometric Vitals Dashboard ────────────────────────────────
    def test_biometric_vitals_engine_and_command(self):
        """Test contactless vitals, frame feed, wellness score, and command."""
        engine = BiometricVitalsEngine()

        # Simulate feeding camera frames
        res = engine.feed_camera_frame(green_mean=128.5, ear=0.32, head_pose=(2.0, 1.0, 0.0))
        assert "heart_rate_bpm" in res
        assert "posture" in res

        vitals = engine.get_current_vitals()
        assert "posture" in vitals
        assert "screen_time_min" in vitals

        score = engine.compute_wellness_score()
        assert score.overall >= 0

        cmd = VitalsCommand()
        cmd.engine = engine

        check_res = cmd.execute("check vitals")
        assert "heart rate" in check_res.lower() or "posture" in check_res.lower()

        score_res = cmd.execute("wellness score")
        assert "wellness score" in score_res.lower()

    # ── Feature 6: Autonomous Voice Pair-Programmer (Code Whisperer) ─────────
    def test_code_whisperer_engine_and_command(self):
        """Test code explanation, generation, rubber ducking, and command."""
        whisperer = CodeWhispererEngine()

        code_sample = "def fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)"
        explanation = whisperer.explain_code(code_sample, language=CodeLanguage.PYTHON)
        assert explanation.explanation is not None
        assert len(explanation.explanation) > 0

        # Test rubber ducking
        duck_response = whisperer.rubber_duck("Why does my recursive fibonacci time out for n=50?")
        assert len(duck_response) > 0

        cmd = CodeWhispererCommand()
        cmd.engine = whisperer

        start_res = cmd.execute("start pair programming")
        assert "pair-programming" in start_res.lower() or "started" in start_res.lower()

        explain_res = cmd.execute("explain this code", entity=code_sample)
        assert "explanation" in explain_res.lower()

        end_res = cmd.execute("end pair programming")
        assert "ended" in end_res.lower()

    # ── Feature 7: Autonomous Task Dispatcher (Multi-Device Mesh) ────────────
    def test_task_dispatcher_and_command(self):
        """Test task queue, DAG decomposition, device registration, and command."""
        dispatcher = AutonomousTaskDispatcher(max_workers=2)
        assert dispatcher.start() is True

        try:
            # Submit simple task
            task = dispatcher.submit_task(
                name="test_ping",
                handler=lambda: {"ping": "pong"},
                priority=TaskPriority.HIGH,
            )
            assert task.task_id is not None

            # Register mesh node
            dev_id = dispatcher.register_device(
                name="test_phone",
                device_type="phone",
                capabilities=["mic", "notifications"],
            )
            assert dev_id is not None

            # Check queue status
            q_status = dispatcher.get_queue_status()
            assert q_status["running"] is True
            assert q_status["mesh_devices"] >= 2  # local + test_phone

            cmd = DispatcherCommand()
            cmd.dispatcher = dispatcher

            status_res = cmd.execute("dispatcher status")
            assert "running" in status_res.lower()

            devices_res = cmd.execute("mesh devices")
            assert "test_phone" in devices_res or "local" in devices_res.lower()

        finally:
            dispatcher.stop()

    # ── Intent Detection & Command Router Mappings ───────────────────────────
    def test_intent_detection_and_routing_frontier_v2(self):
        """Verify IntentDetector detects all 7 new intents and routes correctly."""
        detector = IntentDetector(semantic_enabled=False)

        intents_to_test = [
            ("show avatar", Intent.AVATAR_CONTROL),
            ("start air gestures", Intent.AIR_GESTURE),
            ("enter dream state", Intent.DREAM_STATE),
            ("npu status", Intent.NPU_ACCEL),
            ("check my vitals", Intent.BIOMETRIC_VITALS),
            ("start pair programming", Intent.CODE_WHISPERER),
            ("task dispatcher", Intent.TASK_DISPATCHER),
        ]

        for phrase, expected_intent in intents_to_test:
            detected = detector.detect_intent(phrase)
            assert detected == expected_intent, f"Failed for '{phrase}': got {detected}, expected {expected_intent}"

        # Test router instantiation
        orchestrator = MagicMock()
        router = CommandRouter(orchestrator)
        for expected_intent in [
            Intent.AVATAR_CONTROL,
            Intent.AIR_GESTURE,
            Intent.DREAM_STATE,
            Intent.NPU_ACCEL,
            Intent.BIOMETRIC_VITALS,
            Intent.CODE_WHISPERER,
            Intent.TASK_DISPATCHER,
        ]:
            from sentinel.core.command_router import INTENT_TO_PLUGIN

            assert expected_intent in INTENT_TO_PLUGIN
            plugin_key = INTENT_TO_PLUGIN[expected_intent]
            assert plugin_key in router.plugins, f"Plugin {plugin_key} missing in CommandRouter.plugins"
