import unittest
from sentinel.memory.long_term_memory import LongTermMemory

class TestMemoryExportImport(unittest.TestCase):
    def test_memory_export_import_in_memory(self):
        mem = LongTermMemory()
        # enable in-memory mode explicitly for test isolation
        mem._in_memory = True
        mem._mem_summaries = ["a memory" ]
        mem._mem_facts = ["fact1"]
        data = mem.export_memory()
        self.assertIn('summaries', data)
        self.assertIn('facts', data)
        ok = mem.import_memory({'summaries': ['b memory'], 'facts': ['factB']})
        self.assertTrue(ok)
        self.assertIn('b memory', mem._mem_summaries[-1])
