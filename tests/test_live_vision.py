"""
tests/test_live_vision.py
─────────────────────────
Tests for live spatial vision module.
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLiveVisionImports(unittest.TestCase):
    def test_live_vision_imports(self):
        from sentinel.vision.live_vision import LiveVisionStream
        self.assertTrue(hasattr(LiveVisionStream, 'start'))
        self.assertTrue(hasattr(LiveVisionStream, 'stop'))
        self.assertTrue(hasattr(LiveVisionStream, 'capture_frame'))

    def test_live_vision_command_imports(self):
        from sentinel.commands.live_vision import LiveVisionCommand
        self.assertTrue(hasattr(LiveVisionCommand, 'execute'))


class TestLiveVisionStream(unittest.TestCase):
    def setUp(self):
        from sentinel.vision.live_vision import LiveVisionStream
        self.stream = LiveVisionStream(camera_id=99)

    def test_initial_state(self):
        self.assertFalse(self.stream._running)
        self.assertEqual(self.stream.camera_id, 99)

    def test_get_stats(self):
        stats = self.stream.get_stats()
        self.assertIn("camera_id", stats)
        self.assertIn("fps", stats)


if __name__ == "__main__":
    unittest.main()