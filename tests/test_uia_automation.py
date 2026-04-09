"""
tests/test_uia_automation.py
─────────────────────────────
Tests for UI Automation module.
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestUIAImports(unittest.TestCase):
    def test_uia_automation_imports(self):
        from sentinel.automation.uia_automation import UIAutomation
        self.assertTrue(hasattr(UIAutomation, 'get_foreground_window'))
        self.assertTrue(hasattr(UIAutomation, 'get_window_info'))
        self.assertTrue(hasattr(UIAutomation, 'click_element'))

    def test_uia_command_imports(self):
        from sentinel.commands.ui_automation import UIAutomationCommand
        self.assertTrue(hasattr(UIAutomationCommand, 'execute'))


class TestUIACommand(unittest.TestCase):
    def setUp(self):
        from sentinel.commands.ui_automation import UIAutomationCommand
        self.cmd = UIAutomationCommand()

    def test_window_info_action(self):
        result = self.cmd.execute("window_info")
        self.assertIn("success", result)

    def test_list_windows_action(self):
        result = self.cmd.execute("list_windows")
        self.assertIn("success", result)
        self.assertIn("windows", result)

    def test_unknown_action(self):
        result = self.cmd.execute("unknown_action")
        self.assertFalse(result.get("success"))


if __name__ == "__main__":
    unittest.main()