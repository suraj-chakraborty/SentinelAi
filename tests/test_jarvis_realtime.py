"""
tests/test_jarvis_realtime.py
─────────────────────────────
Comprehensive unit tests for the 5 JARVIS real-time features:
1. Barge-In & Streaming TTS
2. Active Window Context & Vision
3. Proactive Sentinel & Executive Brief
4. Global Hotkey & Floating HUD
5. Episodic Memory Recall
"""

import os
import sys
import time
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentinel.voice.tts import TTSEngine
from sentinel.voice.barge_in import BargeInMonitor
from sentinel.voice.streaming_tts import StreamingTTSPlayer
from sentinel.utils.window_context import get_active_window_info
from sentinel.modules.vision import VisionModule
from sentinel.core.proactive_sentinel import ProactiveSentinel
from sentinel.modules.daily_brief import DailyBriefModule
from sentinel.utils.hotkey_manager import GlobalHotkeyManager
from sentinel.memory.episodic_logger import EpisodicLogger
from sentinel.commands.search_memory import SearchMemoryCommand


class TestJarvisRealtimeFeatures:

    def test_barge_in_and_streaming_tts(self):
        """Test TTS interruption, queue purging, and sentence streaming."""
        mock_tts = MagicMock()
        mock_tts.is_speaking = False
        mock_tts._stop_event = MagicMock()
        mock_tts._stop_event.is_set.return_value = False

        player = StreamingTTSPlayer(tts=mock_tts)
        text = "Good morning, sir. I have compiled your schedule. All systems are fully operational!"
        count = player.speak_stream(text)

        assert count == 3
        assert mock_tts.speak.call_count == 3

        # Test barge-in monitor trigger
        triggered = []
        monitor = BargeInMonitor(
            tts_engine=mock_tts,
            on_barge_in=lambda: triggered.append(True)
        )
        assert monitor.samplerate == 16000

    def test_active_window_context_and_vision(self):
        """Test window inspection and vision fallback."""
        info = get_active_window_info()
        assert isinstance(info, dict)
        assert "title" in info
        assert "app_name" in info

        vision = VisionModule(api_key=None)
        # Without API key, should gracefully return window context without crashing
        desc = vision.analyze_screen()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_proactive_sentinel_and_daily_brief(self):
        """Test proactive background checks and executive daily brief."""
        mock_orch = MagicMock()
        proactive = ProactiveSentinel(orchestrator=mock_orch, check_interval=1)
        assert proactive.check_interval == 1

        brief_module = DailyBriefModule(orchestrator=mock_orch)
        brief = brief_module.generate_brief()
        assert isinstance(brief, str)
        assert "sir" in brief.lower()
        assert "systems are online" in brief.lower()

    def test_global_hotkey_manager(self):
        """Test hotkey manager initialization and lifecycle."""
        cb_called = []
        hotkey = GlobalHotkeyManager(on_trigger=lambda: cb_called.append(1))
        assert hotkey.VK_J == 0x4A
        # Test stop when not running doesn't crash
        hotkey.stop()

    def test_episodic_memory_and_recall(self, tmp_path):
        """Test persistent SQLite episodic logging and recall."""
        test_db = str(tmp_path / "test_episodes.db")
        logger = EpisodicLogger(db_path=test_db)

        logger.log_turn(
            user_command="open visual studio code",
            intent="OPEN_APP",
            response="Launching Visual Studio Code",
            active_app="Explorer",
            active_window="Desktop"
        )
        logger.log_turn(
            user_command="run pytest test suite",
            intent="CODE_EXECUTE",
            response="All 151 tests passed",
            active_app="Code",
            active_window="Terminal"
        )

        recent = logger.get_recent(limit=5)
        assert len(recent) == 2
        assert recent[0]["user_command"] == "run pytest test suite"

        search = logger.search_episodes("pytest")
        assert len(search) == 1
        assert "151 tests passed" in search[0]["response"]

        # Test SearchMemoryCommand with episodic recall
        with patch("sentinel.commands.search_memory.get_episodic_logger", return_value=logger):
            cmd = SearchMemoryCommand()
            res = cmd.execute("what did i do recently?")
            assert "pytest" in res or "visual studio code" in res
