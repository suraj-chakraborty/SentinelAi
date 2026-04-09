"""
tests/test_daemon_manager.py
────────────────────────────
Tests for daemon manager module.
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDaemonManagerImports(unittest.TestCase):
    def test_daemon_manager_imports(self):
        from sentinel.daemons.daemon_manager import DaemonManager
        self.assertTrue(hasattr(DaemonManager, 'start'))
        self.assertTrue(hasattr(DaemonManager, 'stop'))
        self.assertTrue(hasattr(DaemonManager, 'register_task'))

    def test_daemon_command_imports(self):
        from sentinel.commands.daemon_control import DaemonControlCommand
        self.assertTrue(hasattr(DaemonControlCommand, 'execute'))


class TestDaemonManager(unittest.TestCase):
    def setUp(self):
        from sentinel.daemons.daemon_manager import DaemonManager
        self.manager = DaemonManager()

    def test_initial_state(self):
        self.assertFalse(self.manager._running)
        self.assertEqual(len(self.manager._tasks), 0)

    def test_register_task(self):
        def dummy_action():
            return "done"
        
        result = self.manager.register_task(
            task_id="test_task",
            name="Test Task",
            action=dummy_action,
            trigger="manual"
        )
        self.assertTrue(result)
        self.assertIn("test_task", self.manager._tasks)

    def test_list_tasks_empty(self):
        tasks = self.manager.list_tasks()
        self.assertEqual(tasks, [])

    def test_get_stats(self):
        stats = self.manager.get_stats()
        self.assertIn("running", stats)
        self.assertIn("total_tasks", stats)


if __name__ == "__main__":
    unittest.main()