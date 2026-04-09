"""
tests/test_sentinel.py
────────────────────────
Comprehensive unit tests for the SentinelAI core engine.

Coverage across:
  • Preprocessor        (10 cases)
  • IntentDetector      (35 intent cases — one per intent)
  • Slot filling        (8 multi-entity cases)
  • CommandRouter       (10 routing cases)
  • SemanticIntent      (5 mocked cases)
  • LongTermMemory      (5 memory cases)
  • Sandbox             (10 safety cases)
  • AppState            (5 thread-safety cases)
  • PluginSystem        (5 lifecycle cases)
  • QuantumModule       (3 cases)
  • OllamaModule        (3 cases)

IMPORTANT: All tests run without loading heavy ML models, making API calls,
or requiring hardware (microphone, camera). Expensive dependencies are mocked.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disable semantic routing globally so no model is downloaded
os.environ["SENTINEL_SEMANTIC_ROUTING"] = "0"
os.environ["SENTINEL_CODE_SANDBOX_LEVEL"] = "2"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Preprocessor
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        from sentinel.core.preprocessor import preprocess_command
        self.preprocess = preprocess_command

    def test_removes_assistant_name(self):
        self.assertEqual(self.preprocess("Sentinel, open chrome"), "open chrome")

    def test_removes_please(self):
        self.assertEqual(self.preprocess("please open chrome"), "open chrome")

    def test_removes_can_you(self):
        self.assertEqual(self.preprocess("can you shut down"), "shut down")

    def test_removes_punctuation_comma(self):
        result = self.preprocess("hey, open notepad!")
        self.assertNotIn(",", result)
        self.assertNotIn("!", result)

    def test_lowercases(self):
        self.assertEqual(self.preprocess("Open CHROME"), "open chrome")

    def test_strips_whitespace(self):
        self.assertEqual(self.preprocess("   open chrome   "), "open chrome")

    def test_could_you(self):
        self.assertNotIn("could you", self.preprocess("could you show system info"))

    def test_collapse_spaces(self):
        result = self.preprocess("open    chrome")
        self.assertNotIn("  ", result)

    def test_empty_string(self):
        self.assertEqual(self.preprocess(""), "")

    def test_none_input(self):
        self.assertEqual(self.preprocess(None), "")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Intent Detection — one test per intent
# ══════════════════════════════════════════════════════════════════════════════

class TestIntentDetector(unittest.TestCase):

    def setUp(self):
        from sentinel.core.intent_detector import IntentDetector, Intent
        self.detector = IntentDetector(semantic_enabled=False)
        self.Intent = Intent

    def _assert_intent(self, command: str, expected_intent):
        result = self.detector.detect_intent(command)
        self.assertEqual(
            result, expected_intent,
            f"'{command}' → got {result}, expected {expected_intent}",
        )

    # Application control
    def test_open_app(self):         self._assert_intent("open chrome", self.Intent.OPEN_APP)
    def test_launch_app(self):       self._assert_intent("launch notepad", self.Intent.OPEN_APP)
    def test_close_app(self):        self._assert_intent("close discord", self.Intent.CLOSE_APP)
    def test_browse_url(self):       self._assert_intent("go to github.com", self.Intent.BROWSE_URL)

    # System control
    def test_shutdown(self):         self._assert_intent("shutdown computer", self.Intent.SHUTDOWN)
    def test_restart(self):          self._assert_intent("restart computer", self.Intent.RESTART)
    def test_lock_pc(self):          self._assert_intent("lock the screen", self.Intent.LOCK_PC)
    def test_system_status(self):    self._assert_intent("system status", self.Intent.SYSTEM_STATUS)
    def test_volume_up(self):        self._assert_intent("volume up", self.Intent.VOLUME_UP)
    def test_volume_down(self):      self._assert_intent("volume down", self.Intent.VOLUME_DOWN)
    def test_mute(self):             self._assert_intent("mute", self.Intent.VOLUME_MUTE)
    def test_screenshot(self):       self._assert_intent("take a screenshot", self.Intent.TAKE_SCREENSHOT)

    # Media
    def test_play_music(self):       self._assert_intent("play some music", self.Intent.PLAY_MUSIC)
    def test_pause_music(self):      self._assert_intent("pause music", self.Intent.PAUSE_MUSIC)

    # Time management
    def test_set_timer(self):        self._assert_intent("set a timer for 5 minutes", self.Intent.SET_TIMER)
    def test_set_alarm(self):        self._assert_intent("set an alarm", self.Intent.SET_ALARM)
    def test_set_reminder(self):     self._assert_intent("remind me to call John", self.Intent.SET_REMINDER)
    def test_calendar_read(self):    self._assert_intent("what's on my calendar", self.Intent.CALENDAR_READ)
    def test_calendar_add(self):     self._assert_intent("add to calendar", self.Intent.CALENDAR_ADD)

    # Web/search
    def test_get_weather(self):      self._assert_intent("what's the weather", self.Intent.GET_WEATHER)
    def test_get_news(self):         self._assert_intent("top news today", self.Intent.GET_NEWS)
    def test_calculate(self):        self._assert_intent("calculate 15 times 4", self.Intent.CALCULATE)
    def test_translate(self):        self._assert_intent("translate hello to french", self.Intent.TRANSLATE)

    # Email
    def test_send_email(self):       self._assert_intent("send email to boss", self.Intent.SEND_EMAIL)
    def test_read_email(self):       self._assert_intent("read my email", self.Intent.READ_EMAIL)

    # Files
    def test_find_file(self):        self._assert_intent("find file report.pdf", self.Intent.FIND_FILE)
    def test_take_note(self):        self._assert_intent("take a note: buy milk", self.Intent.TAKE_NOTE)
    def test_read_note(self):        self._assert_intent("read my notes", self.Intent.READ_NOTE)

    # KB
    def test_kb_query(self):         self._assert_intent("what do you know about my project", self.Intent.KB_QUERY)
    def test_kb_add(self):           self._assert_intent("save this to memory", self.Intent.KB_ADD)

    # Clipboard
    def test_clipboard_read(self):   self._assert_intent("read clipboard", self.Intent.CLIPBOARD_READ)

    # Agent/Autonomous
    def test_autonomous(self):       self._assert_intent("automate renaming my files", self.Intent.AUTONOMOUS_AGENT)
    def test_deactivate(self):       self._assert_intent("exit sentinel", self.Intent.DEACTIVATE)

    # Search (low priority — must not steal from above)
    def test_search_web(self):       self._assert_intent("search for python tutorials", self.Intent.SEARCH_WEB)

    # Unknown
    def test_unknown(self):
        result = self.detector.detect_intent("xzyzzy nonsensical command foobarbaz")
        self.assertEqual(result, self.Intent.UNKNOWN)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Slot Filling
# ══════════════════════════════════════════════════════════════════════════════

class TestSlotFilling(unittest.TestCase):

    def setUp(self):
        from sentinel.core.intent_detector import IntentDetector, Intent
        self.detector = IntentDetector(semantic_enabled=False)
        self.Intent = Intent

    def test_app_slot(self):
        slots = self.detector.extract_slots("open chrome", self.Intent.OPEN_APP)
        self.assertIn("app", slots)
        self.assertIn("chrome", slots["app"].lower())

    def test_url_slot(self):
        slots = self.detector.extract_slots("go to github.com", self.Intent.BROWSE_URL)
        self.assertIn("url", slots)
        self.assertIn("github", slots["url"])

    def test_duration_slot_minutes(self):
        slots = self.detector.extract_slots("set a timer for 5 minutes", self.Intent.SET_TIMER)
        self.assertIn("duration", slots)
        self.assertIn("5", slots["duration"])

    def test_duration_slot_seconds(self):
        slots = self.detector.extract_slots("timer for 30 seconds", self.Intent.SET_TIMER)
        self.assertIn("duration", slots)

    def test_time_slot(self):
        slots = self.detector.extract_slots("set an alarm for 7am", self.Intent.SET_ALARM)
        self.assertIn("time", slots)

    def test_query_slot(self):
        slots = self.detector.extract_slots("search for python tutorials", self.Intent.SEARCH_WEB)
        self.assertIn("query", slots)
        self.assertIn("python", slots["query"].lower())

    def test_note_content_slot(self):
        slots = self.detector.extract_slots("take a note: buy milk", self.Intent.TAKE_NOTE)
        self.assertIn("note_content", slots)
        self.assertIn("buy milk", slots["note_content"].lower())

    def test_volume_percent_slot(self):
        slots = self.detector.extract_slots("set volume to 60 percent", self.Intent.VOLUME_UP)
        self.assertIn("volume_level", slots)
        self.assertIn("60", slots["volume_level"])


# ══════════════════════════════════════════════════════════════════════════════
# 4. Command Router
# ══════════════════════════════════════════════════════════════════════════════

class TestCommandRouter(unittest.TestCase):

    def setUp(self):
        self.mock_orc = MagicMock()
        from sentinel.core.command_router import CommandRouter
        from sentinel.core.intent_detector import IntentDetector
        self.router = CommandRouter(self.mock_orc)
        self.router.detector = IntentDetector(semantic_enabled=False)

    def _mock_plugin(self, name, response):
        m = MagicMock()
        m.execute.return_value = response
        self.router.plugins[name] = m
        return m

    def test_open_app_routes_to_plugin(self):
        mock = self._mock_plugin("open_app", "Opening chrome")
        result = self.router.run_command("open chrome")
        self.assertEqual(result, "Opening chrome")
        mock.execute.assert_called_once()

    def test_close_app_routes_to_plugin(self):
        mock = self._mock_plugin("close_app", "Closed notepad.")
        result = self.router.run_command("close notepad")
        self.assertEqual(result, "Closed notepad.")

    def test_system_control_shutdown(self):
        mock = self._mock_plugin("system_control", "Shutting down")
        result = self.router.run_command("shutdown computer")
        self.assertEqual(result, "Shutting down")

    def test_system_control_restart(self):
        mock = self._mock_plugin("system_control", "Restarting")
        result = self.router.run_command("restart computer")
        self.assertEqual(result, "Restarting")

    def test_ai_agent_fallback_for_complex(self):
        mock = self._mock_plugin("ai_agent", "Executing autonomous task")
        result = self.router.run_command("organize my downloads folder by file type and date")
        self.assertEqual(result, "Executing autonomous task")

    def test_deactivate_routes(self):
        mock = self._mock_plugin("deactivate", "Shutting down Sentinel")
        result = self.router.run_command("exit sentinel")
        self.assertEqual(result, "Shutting down Sentinel")

    def test_empty_command_returns_none(self):
        result = self.router.run_command("")
        self.assertIsNone(result)

    def test_preprocessor_applied(self):
        mock = self._mock_plugin("open_app", "Opening chrome")
        result = self.router.run_command("Sentinel, please open chrome!")
        self.assertEqual(result, "Opening chrome")

    def test_context_updated(self):
        self._mock_plugin("open_app", "Opening chrome")
        self.router.run_command("open chrome")
        # Context manager should have been called
        ctx = self.router.context
        self.assertIsNotNone(ctx)

    def test_none_command(self):
        result = self.router.run_command(None)
        self.assertIsNone(result)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Semantic Intent (mocked)
# ══════════════════════════════════════════════════════════════════════════════

class TestSemanticIntent(unittest.TestCase):

    @patch("sentinel.core.semantic_intent.get_semantic_classifier")
    def test_semantic_used_as_fallback(self, mock_get_clf):
        mock_clf = MagicMock()
        mock_clf.predict.return_value = ("open_app", 0.70)
        mock_get_clf.return_value = mock_clf

        from sentinel.core.intent_detector import IntentDetector, Intent
        detector = IntentDetector(semantic_enabled=True)
        detector._semantic_clf = mock_clf

        intent = detector.detect_intent("zzrootzonic completely made up nonsense")
        self.assertEqual(intent, Intent.OPEN_APP)
        mock_clf.predict.assert_called_once()

    def test_semantic_disabled_returns_unknown(self):
        from sentinel.core.intent_detector import IntentDetector, Intent
        detector = IntentDetector(semantic_enabled=False)
        intent = detector.detect_intent("completely made up nonsense")
        self.assertEqual(intent, Intent.UNKNOWN)

    def test_confidence_below_threshold_returns_unknown(self):
        mock_clf = MagicMock()
        mock_clf.predict.return_value = (None, 0.20)   # below threshold → returns (None, score)

        from sentinel.core.intent_detector import IntentDetector, Intent
        detector = IntentDetector(semantic_enabled=True)
        detector._semantic_clf = mock_clf

        intent = detector.detect_intent("zzrootzonic nonsense xyrqz")
        # keyword and fuzzy miss; semantic returns None → UNKNOWN
        self.assertEqual(intent, Intent.UNKNOWN)

    def test_detect_with_confidence_returns_tuple(self):
        from sentinel.core.intent_detector import IntentDetector, Intent
        detector = IntentDetector(semantic_enabled=False)
        intent, conf = detector.detect_with_confidence("open chrome")
        self.assertEqual(intent, Intent.OPEN_APP)
        self.assertIsInstance(conf, float)
        self.assertGreater(conf, 0.0)

    def test_keyword_wins_over_semantic(self):
        """Keyword match must return before semantic is consulted."""
        mock_clf = MagicMock()
        mock_clf.predict.return_value = ("shutdown", 0.9)

        from sentinel.core.intent_detector import IntentDetector, Intent
        detector = IntentDetector(semantic_enabled=True)
        detector._semantic_clf = mock_clf

        intent = detector.detect_intent("open chrome")   # clear keyword match
        self.assertEqual(intent, Intent.OPEN_APP)
        mock_clf.predict.assert_not_called()             # semantic never reached


# ══════════════════════════════════════════════════════════════════════════════
# 6. Sandbox — AST safety
# ══════════════════════════════════════════════════════════════════════════════

class TestSandbox(unittest.TestCase):

    def setUp(self):
        from sentinel.security.sandbox import Sandbox
        self.Sandbox = Sandbox

    def test_safe_code_passes(self):
        ok, output = self.Sandbox.execute("print(2 + 2)", level=2)
        self.assertTrue(ok)
        self.assertIn("4", output)

    def test_import_os_blocked(self):
        ok, msg = self.Sandbox.execute("import os\nprint(os.getcwd())", level=2)
        self.assertFalse(ok)
        self.assertIn("os", msg.lower())

    def test_import_subprocess_blocked(self):
        ok, msg = self.Sandbox.execute("import subprocess", level=2)
        self.assertFalse(ok)

    def test_eval_blocked(self):
        ok, msg = self.Sandbox.execute("eval('1+1')", level=2)
        self.assertFalse(ok)

    def test_exec_blocked(self):
        ok, msg = self.Sandbox.execute("exec('x=1')", level=2)
        self.assertFalse(ok)

    def test_dunder_import_blocked(self):
        ok, msg = self.Sandbox.execute("__import__('os').system('echo hi')", level=2)
        self.assertFalse(ok)

    def test_import_ctypes_blocked(self):
        ok, msg = self.Sandbox.execute("import ctypes", level=2)
        self.assertFalse(ok)

    def test_empty_code_fails(self):
        ok, msg = self.Sandbox.execute("", level=2)
        self.assertFalse(ok)

    def test_syntax_error_fails(self):
        ok, msg = self.Sandbox.execute("def foo(: pass", level=2)
        self.assertFalse(ok)

    def test_safety_report(self):
        report = self.Sandbox.get_safety_report("print('hello')")
        self.assertTrue(report["safe"])
        self.assertTrue(report["syntax_ok"])


# ══════════════════════════════════════════════════════════════════════════════
# 7. AppState — thread safety
# ══════════════════════════════════════════════════════════════════════════════

class TestAppState(unittest.TestCase):

    def setUp(self):
        from sentinel.app.state import reset_state, get_state
        reset_state()
        self.get_state = get_state

    def test_singleton(self):
        from sentinel.app.state import get_state
        s1 = get_state()
        s2 = get_state()
        self.assertIs(s1, s2)

    def test_block_listening(self):
        state = self.get_state()
        state.block_listening(5)
        self.assertTrue(state.listening_blocked)
        state.unblock_listening()
        self.assertFalse(state.listening_blocked)

    def test_agent_active(self):
        state = self.get_state()
        self.assertFalse(state.agent_active)
        state.agent_active = True
        self.assertTrue(state.agent_active)

    def test_brain_status(self):
        state = self.get_state()
        state.brain_status = "Active"
        self.assertEqual(state.brain_status, "Active")

    def test_concurrent_writes(self):
        """Multiple threads must not corrupt state."""
        import threading
        state = self.get_state()
        errors = []

        def _writer():
            for _ in range(1000):
                try:
                    state.agent_active = True
                    state.agent_active = False
                    state.brain_status = "OK"
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=_writer) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(len(errors), 0, f"Thread safety violations: {errors}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. PluginSystem lifecycle
# ══════════════════════════════════════════════════════════════════════════════

class TestPluginSystem(unittest.TestCase):

    def setUp(self):
        from sentinel.core.plugin_system import PluginSystem
        self.ps = PluginSystem(orchestrator=MagicMock())

    def test_empty_on_init(self):
        self.assertEqual(len(self.ps), 0)

    def test_is_loaded_false(self):
        self.assertFalse(self.ps.is_loaded("NonExistent"))

    def test_list_plugins_empty(self):
        self.assertEqual(self.ps.list_plugins(), [])

    def test_dispatch_returns_none_when_empty(self):
        result = self.ps.dispatch("open chrome")
        self.assertIsNone(result)

    def test_load_bad_path_returns_false(self):
        result = self.ps.load_plugin("/nonexistent/path/to/plugin")
        self.assertFalse(result)


# ══════════════════════════════════════════════════════════════════════════════
# 9. QuantumModule
# ══════════════════════════════════════════════════════════════════════════════

class TestQuantumModule(unittest.TestCase):

    def setUp(self):
        from sentinel.modules.quantum_module import QuantumModule
        self.qm = QuantumModule()   # no API token → local solver only

    def test_local_solve_returns_string(self):
        result = self.qm.solve_optimization("route 5 cities")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 20)

    def test_local_solve_contains_algorithm_name(self):
        result = self.qm.solve_optimization("any problem")
        self.assertIn("Annealing", result)

    def test_no_api_token_uses_local(self):
        result = self.qm.solve_optimization("problem", llm_callback=None)
        # Without token, must fall back to local solver
        self.assertNotIn("API token", result)


# ══════════════════════════════════════════════════════════════════════════════
# 10. OllamaModule
# ══════════════════════════════════════════════════════════════════════════════

class TestOllamaModule(unittest.TestCase):

    def setUp(self):
        from sentinel.modules.ollama_module import OllamaModule
        self.om = OllamaModule()

    def test_model_selected(self):
        self.assertIn(self.om.model, ("llama3", "tinyllama", "llama3.2"))

    def test_is_available_returns_bool(self):
        # We don't assert True/False — just that it doesn't throw
        result = self.om.is_available()
        self.assertIsInstance(result, bool)

    def test_generate_when_unavailable(self):
        """If Ollama is offline, generate should return a human-readable error."""
        self.om._available_cache = False
        self.om._last_check = float("inf")
        result = self.om.generate("Hello world")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 5)


# ══════════════════════════════════════════════════════════════════════════════
# 11. LongTermMemory (no ChromaDB required — mocked)
# ══════════════════════════════════════════════════════════════════════════════

class TestLongTermMemory(unittest.TestCase):

    def setUp(self):
        from sentinel.memory.long_term_memory import LongTermMemory
        # Patch ChromaDB to avoid actual DB creation
        with patch("sentinel.memory.long_term_memory.LongTermMemory._init_db"):
            self.ltm = LongTermMemory(llm_callback=MagicMock(return_value="• Bullet summary"))

    def test_stats_when_unavailable(self):
        stats = self.ltm.get_stats()
        self.assertIsInstance(stats, dict)

    def test_inject_past_context_when_unavailable(self):
        result = self.ltm.inject_past_context("test query")
        self.assertIsInstance(result, str)

    def test_store_fact_when_unavailable(self):
        result = self.ltm.store_fact("I prefer dark mode")
        self.assertIsInstance(result, bool)

    def test_recall_facts_when_unavailable(self):
        result = self.ltm.recall_facts("theme preference")
        self.assertIsInstance(result, str)

    def test_on_new_turn_doesnt_crash(self):
        mock_conv = MagicMock()
        mock_conv.get_recent_context.return_value = ""
        # Should not raise even when unavailable
        for _ in range(25):
            self.ltm.on_new_turn(mock_conv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
