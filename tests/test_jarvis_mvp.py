import unittest
from sentinel.core.jarvis_persona import JarvisPersona
from sentinel.core.retrieval import RetrievalContext, PromptBuilder


class TestJarvisMVP(unittest.TestCase):
    def test_jarvis_persona_prompt_integration(self):
        jp = JarvisPersona(tone="calm")
        rc = RetrievalContext(memory_context="Memory: test", retrieved_context="Retrieved: facts", knowledge_context="KB: notes", current_task="What should I do?")
        rc.persona = jp.get_prompt_schip()
        pb = PromptBuilder(rc)
        prompt = pb.build()
        self.assertIn("Jarvis", prompt)  # persona line should be present
        self.assertIn("Memory: test", prompt)
        self.assertIn("Retrieved: facts", prompt)
        self.assertIn("KB: notes", prompt)
        self.assertIn("What should I do?", prompt)

    def test_planner_basic(self):
        from sentinel.core.planner import Planner
        p = Planner()
        steps = p.plan("organize downloads")
        self.assertTrue(len(steps) >= 1)
