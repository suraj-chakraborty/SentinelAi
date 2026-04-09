import unittest
from unittest.mock import MagicMock


class TestRetrievalModule(unittest.TestCase):
    def test_retrieve_calls_knowledge_base(self):
        kb = MagicMock()
        kb.query_knowledge.return_value = "Some relevant knowledge"
        from sentinel.modules.retrieval_module import RetrievalModule
        rm = RetrievalModule(knowledge_base=kb)
        out = rm.retrieve("what is this about?")
        self.assertEqual(out, "Some relevant knowledge")
