"""
tests/test_screen_memory.py
────────────────────────────
Tests for semantic screen memory module.
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestScreenMemoryImports(unittest.TestCase):
    def test_screen_capture_imports(self):
        from sentinel.memory.screen_capture import ScreenCapture
        self.assertTrue(hasattr(ScreenCapture, 'capture_full_screen'))
        self.assertTrue(hasattr(ScreenCapture, 'extract_text'))

    def test_screen_memory_db_imports(self):
        from sentinel.memory.screen_memory_db import ScreenMemoryDB
        self.assertTrue(hasattr(ScreenMemoryDB, 'add_screen_memory'))
        self.assertTrue(hasattr(ScreenMemoryDB, 'search'))

    def test_screen_indexer_imports(self):
        from sentinel.memory.screen_indexer import ScreenIndexer
        self.assertTrue(hasattr(ScreenIndexer, 'start'))
        self.assertTrue(hasattr(ScreenIndexer, 'stop'))
        self.assertTrue(hasattr(ScreenIndexer, 'search'))

    def test_search_memory_command_imports(self):
        from sentinel.commands.search_memory import SearchMemoryCommand
        self.assertTrue(hasattr(SearchMemoryCommand, 'execute'))


class TestScreenMemoryDB(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        from sentinel.memory.screen_memory_db import ScreenMemoryDB
        self.db = ScreenMemoryDB(persist_directory=self.temp_dir)

    def test_add_and_search(self):
        screen_id = self.db.add_screen_memory(
            text="This is a test screenshot with Python code",
            metadata={"source": "test"}
        )
        self.assertIsNotNone(screen_id)
        
        results = self.db.search("Python code", n_results=1)
        self.assertGreater(len(results), 0)
        self.assertIn("Python", results[0]["text"])

    def test_get_recent(self):
        self.db.add_screen_memory(text="First screen", metadata={"source": "test"})
        self.db.add_screen_memory(text="Second screen", metadata={"source": "test"})
        
        recent = self.db.get_recent_screens(limit=2)
        self.assertEqual(len(recent), 2)

    def test_get_stats(self):
        self.db.add_screen_memory(text="Test screen", metadata={"source": "test"})
        stats = self.db.get_stats()
        self.assertIn("total_screens", stats)
        self.assertEqual(stats["total_screens"], 1)


if __name__ == "__main__":
    unittest.main()