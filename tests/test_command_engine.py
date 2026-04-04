import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add the project root to sys.path for imports to work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel.core.command_router import CommandRouter
from sentinel.core.preprocessor import preprocess_command
from sentinel.core.intent_detector import IntentDetector, Intent

class TestCommandEngine(unittest.TestCase):

    def setUp(self):
        # Create a mock orchestrator to pass to the router
        self.mock_orchestrator = MagicMock()
        self.router = CommandRouter(self.mock_orchestrator)
        # Avoid loading sentence-transformers during tests
        self.router.detector = IntentDetector(semantic_enabled=False)

    def test_preprocessor(self):
        """Test if filler words and punctuation are removed correctly."""
        test_cases = [
            ("Sentinel, please open Chrome!", "open chrome"),
            ("Hey assistant, can you shut down?", "shut down"),
            ("Could you show me system info", "system info"),
            ("  organize my   downloads folder. ", "organize my downloads folder")
        ]
        for input_text, expected_output in test_cases:
            with self.subTest(input_text=input_text):
                self.assertEqual(preprocess_command(input_text), expected_output)

    def test_intent_detection_open_app(self):
        """Test mapping 'open' commands to OPEN_APP intent."""
        detector = IntentDetector(semantic_enabled=False)
        intent = detector.detect_intent("open chrome")
        entity = detector.extract_entity("open chrome", intent)
        self.assertEqual(intent, Intent.OPEN_APP)
        self.assertEqual(entity, "chrome")

    def test_intent_detection_system_control(self):
        """Test mapping system commands to shutdown / restart intents."""
        detector = IntentDetector(semantic_enabled=False)
        intent = detector.detect_intent("shutdown")
        self.assertEqual(intent, Intent.SHUTDOWN)

        intent = detector.detect_intent("restart the system")
        self.assertEqual(intent, Intent.RESTART)

    def test_router_dispatches_system_control_plugin(self):
        """Shutdown/restart intents must map to the system_control plugin module."""
        self.router.plugins["system_control"] = MagicMock()
        self.router.plugins["system_control"].execute.return_value = "ok-shutdown"

        response = self.router.run_command("shutdown")
        self.assertEqual(response, "ok-shutdown")
        self.router.plugins["system_control"].execute.assert_called_once()

    def test_router_dispatches_close_app_plugin(self):
        """Close commands route to the close_app plugin."""
        self.router.plugins["close_app"] = MagicMock()
        self.router.plugins["close_app"].execute.return_value = "Closed notepad."

        response = self.router.run_command("close notepad")
        self.assertEqual(response, "Closed notepad.")
        self.router.plugins["close_app"].execute.assert_called_once()

    def test_router_dispatch(self):
        """Test if the router correctly dispatches to plugins."""
        # We'll mock the specific plugin call inside run_command
        # and ensure the plugin exists in the mock router's plugins dict
        self.router.plugins["open_app"] = MagicMock()
        self.router.plugins["open_app"].execute.return_value = "Opening chrome"
        
        response = self.router.run_command("open chrome")
        self.assertEqual(response, "Opening chrome")

    def test_fallback_to_ai(self):
        """Test if complex tasks are routed to the AI Agent plugin."""
        self.router.plugins["ai_agent"] = MagicMock()
        self.router.plugins["ai_agent"].execute.return_value = "Executing autonomous task"
        
        response = self.router.run_command("organize my downloads folder by file type")
        self.assertEqual(response, "Executing autonomous task")

    @patch("sentinel.core.semantic_intent.get_semantic_classifier")
    def test_semantic_layer_when_keywords_miss(self, mock_get_clf):
        """Paraphrases with no keyword hit can be classified via embeddings (mocked)."""
        mock_clf = MagicMock()
        mock_clf.predict.return_value = ("open_app", 0.55)
        mock_get_clf.return_value = mock_clf

        detector = IntentDetector(semantic_enabled=True)
        intent = detector.detect_intent("zzrootzonic flibbledee nonsensecommand")
        self.assertEqual(intent, Intent.OPEN_APP)
        mock_clf.predict.assert_called_once()

if __name__ == "__main__":
    unittest.main()
