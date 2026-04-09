import unittest

class TestPromptBuilder(unittest.TestCase):
    def test_build_basic(self):
        from sentinel.core.retrieval import RetrievalContext, PromptBuilder
        rc = RetrievalContext(memory_context="Memory here", retrieved_context="Data", knowledge_context="KB", current_task="do something")
        pb = PromptBuilder(rc)
        s = pb.build()
        self.assertIn("Memory here", s)
        self.assertIn("[Retrieved]", s)
        self.assertIn("[Knowledge]", s)
        self.assertIn("[Current Task]", s)
