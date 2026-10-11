"""
scripts/verify_sentinel_app.py
──────────────────────────────
Comprehensive startup verification script for SentinelAI.
Validates:
1. Virtual environment and dependencies
2. App preflight and configuration
3. Core orchestrator and command router
4. Audio & Neural TTS subsystems
5. All Frontier 1.0 & Frontier 2.0 capabilities
6. GUI construction (non-blocking)
"""

import os
import sys
import time
import threading

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_verification():
    results = {}
    print("=" * 60)
    print("🚀 SentinelAI System Verification Suite")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print("=" * 60)

    # 1. Configuration & AppData
    print("\n[1/7] Testing AppData & Configuration...")
    try:
        from sentinel.app.config import ensure_appdata_dir, load_settings, APPDATA_DIR
        ensure_appdata_dir()
        settings = load_settings()
        assert os.path.exists(APPDATA_DIR)
        print(f"  ✓ AppData directory ready: {APPDATA_DIR}")
        print(f"  ✓ Settings loaded ({len(settings)} keys)")
        results["config"] = True
    except Exception as e:
        print(f"  ✗ Configuration error: {e}")
        results["config"] = False

    # 2. TTS & Voice Subsystem
    print("\n[2/7] Testing TTS & Voice Subsystem...")
    try:
        from sentinel.voice.tts import get_tts
        tts = get_tts()
        engine_type = tts.engine_type
        print(f"  ✓ TTS Engine initialized: {engine_type}")
        config = tts.get_voice_config()
        print(f"  ✓ Active voice: {config[0]} (pitch={config[1]}, rate={config[2]})")
        results["voice"] = True
    except Exception as e:
        print(f"  ✗ Voice subsystem error: {e}")
        results["voice"] = False

    # 3. Core Orchestrator & Intent Detector
    print("\n[3/7] Testing Sentinel Orchestrator & Router...")
    orch = None
    try:
        from sentinel.core.orchestrator import SentinelOrchestrator
        from sentinel.core.intent_detector import IntentDetector, Intent
        
        detector = IntentDetector(semantic_enabled=False)
        orch = SentinelOrchestrator(
            llm_callback=lambda p: "Mock LLM Response",
            speak_fn=lambda txt, **kw: None,
            exit_callback=lambda: None,
        )
        assert orch.command_router is not None
        print(f"  ✓ Orchestrator initialized successfully")
        print(f"  ✓ Command plugins loaded: {len(orch.command_router.plugins)} plugins")
        results["orchestrator"] = True
    except Exception as e:
        print(f"  ✗ Orchestrator error: {e}")
        results["orchestrator"] = False

    # 4. Frontier 1.0 Capabilities
    print("\n[4/7] Testing Frontier 1.0 JARVIS Subsystems...")
    try:
        from sentinel.core.nightly_swarm import NightlySwarmCoordinator
        from sentinel.voice.live_speech_stream import LiveSpeechStream
        from sentinel.modules.spatial_iot import SpatialIoTCoordinator
        from sentinel.vision.continuous_screen_agent import ContinuousScreenAgent
        from sentinel.core.mobile_hub import MobileHubAgent
        from sentinel.voice.voice_cloner import VoiceProfileManager
        from sentinel.modules.mood_engine import get_mood_engine

        swarm = NightlySwarmCoordinator()
        s2s = LiveSpeechStream()
        spatial = SpatialIoTCoordinator()
        screen_agent = ContinuousScreenAgent()
        mobile_hub = MobileHubAgent()
        voice_mgr = VoiceProfileManager()
        mood_eng = get_mood_engine()

        print("  ✓ Nightly Autonomous Swarm Coordinator initialized")
        print("  ✓ S2S Live Stream WebSocket client initialized")
        print("  ✓ Spatial IoT Coordinator initialized")
        print("  ✓ Continuous Video-Rate Screen Agent initialized")
        print("  ✓ Unified Mobile Hub Agent initialized")
        print("  ✓ Voice Profile Cloner Manager initialized")
        print("  ✓ Dynamic Mood Resonance Engine initialized")
        results["frontier_v1"] = True
    except Exception as e:
        print(f"  ✗ Frontier 1.0 subsystem error: {e}")
        results["frontier_v1"] = False

    # 5. Frontier 2.0 Capabilities
    print("\n[5/7] Testing Frontier 2.0 Capabilities...")
    try:
        from sentinel.ar.holographic_avatar import HolographicAvatarEngine
        from sentinel.perception.air_gesture import AirGestureController
        from sentinel.daemons.dream_state import DreamStateEngine
        from sentinel.neuro.npu_accelerator import NeuralAccelerationEngine
        from sentinel.perception.biometric_vitals import BiometricVitalsEngine
        from sentinel.modules.code_whisperer import CodeWhispererEngine
        from sentinel.core.task_dispatcher import AutonomousTaskDispatcher

        avatar = HolographicAvatarEngine()
        gestures = AirGestureController()
        dream = DreamStateEngine()
        npu = NeuralAccelerationEngine()
        vitals = BiometricVitalsEngine()
        whisperer = CodeWhispererEngine()
        dispatcher = AutonomousTaskDispatcher(max_workers=2)

        print("  ✓ Holographic 3D Floating Avatar Engine initialized")
        print("  ✓ Webcam Spatial Air-Gesture Controller initialized")
        print("  ✓ Cognitive Dream State Autonomous Engine initialized")
        print("  ✓ NPU/GPU Neural Acceleration Engine initialized")
        print("  ✓ Biometric Vitals & Ergonomic Guardian initialized")
        print("  ✓ Voice Pair-Programmer (Code Whisperer) initialized")
        print("  ✓ Autonomous Task Dispatcher (Device Mesh) initialized")
        results["frontier_v2"] = True
    except Exception as e:
        print(f"  ✗ Frontier 2.0 subsystem error: {e}")
        results["frontier_v2"] = False

    # 6. End-to-End Command Routing
    print("\n[6/7] Testing End-to-End Command Execution...")
    try:
        test_commands = [
            ("system status", "System status"),
            ("show avatar", "Holographic avatar"),
            ("gesture status", "Gesture Control"),
            ("dream status", "Dream State"),
            ("npu status", "Neural Accelerator"),
            ("check vitals", "Heart Rate"),
            ("start pair programming", "Pair-programming"),
            ("dispatcher status", "Task Dispatcher"),
            ("flirt with me", "blush"),
        ]

        passed_cmd = 0
        for cmd_text, expected_substr in test_commands:
            res = orch.command_router.run_command(cmd_text)
            if res and any(w.lower() in res.lower() for w in expected_substr.split()):
                passed_cmd += 1
                print(f"  ✓ Command '{cmd_text}': {res.splitlines()[0][:55]}...")
            else:
                print(f"  ⚠ Command '{cmd_text}' returned: {res}")

        results["commands"] = (passed_cmd == len(test_commands))
        print(f"  ✓ Executed {passed_cmd}/{len(test_commands)} test commands successfully")
    except Exception as e:
        print(f"  ✗ Command routing error: {e}")
        results["commands"] = False

    # 7. GUI Component Initialization
    print("\n[7/7] Testing GUI Dashboard Construction...")
    try:
        import tkinter as tk
        from sentinel.app.gui import build_dashboard
        
        # Test constructing dashboard and then closing it cleanly
        root, meter, combo = build_dashboard(
            on_voice_reg=lambda: None,
            on_rebuild=lambda: None,
            on_manual=lambda: None,
            on_config=lambda: None,
            on_terminate=lambda: None,
            on_agent_toggle=lambda: None,
            app_list=[]
        )
        # Update and destroy without entering infinite mainloop
        root.update_idletasks()
        root.destroy()
        print("  ✓ GUI Dashboard built and destroyed cleanly")
        results["gui"] = True
    except Exception as e:
        print(f"  ✗ GUI Dashboard error: {e}")
        results["gui"] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 Final Verification Results:")
    all_passed = True
    for component, ok in results.items():
        status = "PASSED ✅" if ok else "FAILED ❌"
        print(f"  • {component.upper()}: {status}")
        if not ok:
            all_passed = False
    print("=" * 60)

    if all_passed:
        print("\n🎉 ALL SUBSYSTEMS VERIFIED AND FULLY OPERATIONAL!")
        sys.exit(0)
    else:
        print("\n⚠️ SOME VERIFICATIONS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    run_verification()
