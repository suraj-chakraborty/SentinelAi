"""
tests/test_advanced_jarvis_suite.py
───────────────────────────────────
Unit tests for the 6 Advanced Real-Time JARVIS capabilities:
1. Terminal & IDE Watcher + Git Copilot
2. System Audio Meeting Co-Pilot
3. Webcam Desk Presence & Gaze-Wake
4. Deep Autonomous Desktop RPA
5. Live Autonomous Web Browser Agent
6. Focus Shield & Ergonomic Health Guardian
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentinel.automation.ide_watcher import IDEWatcher
from sentinel.commands.git_copilot import GitCopilotCommand
from sentinel.commands.meeting_command import MeetingCommand
from sentinel.perception.gaze_presence import GazePresenceDetector
from sentinel.automation.deep_rpa import DeepRPA
from sentinel.commands.ui_automation import UIAutomationCommand
from sentinel.commands.web_automation import WebAutomation
from sentinel.security.focus_guardian import FocusGuardian
from sentinel.commands.focus_command import FocusCommand
from sentinel.core.intent_detector import IntentDetector, Intent


class TestAdvancedJarvisCapabilities:

    def test_ide_watcher_and_git_copilot(self):
        """Test stack trace analysis and git copilot commands."""
        watcher = IDEWatcher()
        trace = '''
Traceback (most recent call last):
  File "sentinel/core/command_router.py", line 42, in run_command
    return self.plugins[plugin_key].execute()
KeyError: 'unknown_plugin'
'''
        res = watcher.analyze_stack_trace(trace)
        assert res["file"] == "sentinel/core/command_router.py"
        assert res["line"] == "42"
        assert "KeyError" in res["error_type"]

        copilot = GitCopilotCommand()
        with patch.object(watcher, "git_smart_commit", return_value={"success": True, "commit_message": "feat: test"}):
            copilot.watcher = watcher
            msg = copilot.execute("commit changes with message test")
            assert "Committed successfully" in msg

    def test_meeting_copilot(self):
        """Test meeting lifecycle and action item command."""
        mock_surrogate = MagicMock()
        mock_surrogate.start_meeting.return_value = True
        mock_surrogate.end_meeting.return_value = MagicMock(action_items=["Review PR #4"])
        mock_surrogate._action_items = ["Deploy release"]

        cmd = MeetingCommand()
        cmd._surrogate = mock_surrogate

        start_res = cmd.execute("start meeting notes")
        assert "Meeting co-pilot active" in start_res

        summary_res = cmd.execute("meeting action items")
        assert "Deploy release" in summary_res

        end_res = cmd.execute("end meeting")
        assert "Captured 1 action item" in end_res

    def test_gaze_presence_detector(self):
        """Test desk presence detector and lock API."""
        detector = GazePresenceDetector(camera_id=999, absence_timeout=10.0)
        assert detector.is_present is True
        assert detector.is_gaze_engaged is False
        detector.stop()

    def test_deep_rpa(self, tmp_path):
        """Test window snapping, spreadsheet append, and directory sorting."""
        rpa = DeepRPA()

        # 1. Spreadsheet append
        csv_file = str(tmp_path / "expenses.csv")
        ok = rpa.append_spreadsheet_row(csv_file, ["2026-10-10", "Coffee", 4.50])
        assert ok is True
        assert os.path.exists(csv_file)

        # 2. Directory sorting
        test_dir = str(tmp_path / "downloads")
        os.makedirs(test_dir, exist_ok=True)
        with open(os.path.join(test_dir, "invoice.pdf"), "w") as f: f.write("test")
        with open(os.path.join(test_dir, "setup.exe"), "w") as f: f.write("test")

        sort_res = rpa.organize_directory(test_dir)
        assert sort_res["success"] is True
        assert sort_res["files_organized"] == 2
        assert os.path.exists(os.path.join(test_dir, "Documents", "invoice.pdf"))
        assert os.path.exists(os.path.join(test_dir, "Installers", "setup.exe"))

        # 3. UI Automation command
        uia_cmd = UIAutomationCommand()
        snap_res = uia_cmd.execute("snap window to left")
        assert "left" in snap_res.lower()

    def test_web_automation_command(self):
        """Test web navigation and autonomous search dispatch."""
        mock_orch = MagicMock()
        mock_orch.playwright_agent.perform_autonomous_task.return_value = "Extracted lowest non-stop fare: $420"

        web = WebAutomation(orchestrator=mock_orch)
        res = web.execute("check flights to London")
        assert "$420" in res

    def test_focus_guardian_and_command(self):
        """Test Focus Shield activation and distraction handling."""
        guardian = FocusGuardian()
        res = guardian.start_focus_mode(duration_minutes=25)
        assert res["success"] is True
        assert guardian.in_focus_mode is True

        cmd = FocusCommand()
        with patch("sentinel.commands.focus_command.get_focus_guardian", return_value=guardian):
            status_res = cmd.execute("how long have i worked")
            assert "minutes" in status_res

            stop_res = cmd.execute("stop focus mode")
            assert "deactivated" in stop_res.lower()

    def test_intent_detection_and_routing(self):
        """Test that all 6 advanced features detect correct intents."""
        detector = IntentDetector()

        assert detector.detect_intent("commit changes to git") == Intent.GIT_COPILOT
        assert detector.detect_intent("start meeting notes") == Intent.MEETING_COPILOT
        assert detector.detect_intent("start focus mode for 30 minutes") == Intent.FOCUS_SHIELD
        assert detector.detect_intent("snap window to left") == Intent.UI_AUTOMATION
        assert detector.detect_intent("scrape prices from the website") == Intent.WEB_AUTOMATION
