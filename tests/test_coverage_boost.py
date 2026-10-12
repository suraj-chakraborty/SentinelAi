"""
tests/test_coverage_boost.py
─────────────────────────────
Comprehensive unit tests for SentinelAI core modules to guarantee >40% test coverage in CI.
"""

import json
import logging
import os
import tempfile
import unittest
from unittest.mock import patch

from sentinel.app.intelligence import gemini_generate, search_the_web
from sentinel.app.monitor import HealthMonitor
from sentinel.app.registry import (
    load_apps_index,
    load_mru,
    normalize_name,
    save_apps_index,
    save_mru,
)
from sentinel.app.secrets import (
    AESGCM,
    _keystream,
    _pbkdf,
    check_master_exists,
    decrypt_data,
    encrypt_data,
    fetch_secret,
    set_master_password,
    store_secret,
    verify_master,
)
from sentinel.core.conversation import ConversationManager
from sentinel.core.exceptions import (
    ActionExecutionError,
    HardwareError,
    LLMServiceError,
    SandboxError,
    SentinelError,
    VaultError,
)
from sentinel.core.logger import ConsoleFormatter, JSONFormatter, setup_logging
from sentinel.core.orchestrator import SentinelOrchestrator
from sentinel.core.user_profiles import UserProfileManager
from sentinel.models.user_profile import UserProfile
from sentinel.modules.calendar import CalendarModule
from sentinel.modules.coding import CodingModule
from sentinel.modules.system_monitor import SystemMonitor
from sentinel.voice.enhanced_tts import EnhancedTTS
from sentinel.voice.fast_voice import FastVoiceHandler


class TestUserProfileManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        self.tmp.close()
        self.mgr = UserProfileManager(db_path=self.tmp.name)

    def tearDown(self):
        if hasattr(self.mgr, "_conn"):
            self.mgr._conn.close()
        if os.path.exists(self.tmp.name):
            try:
                os.remove(self.tmp.name)
            except Exception:
                pass

    def test_set_and_get_profile(self):
        prof = self.mgr.set_profile(
            user_id="user_alpha",
            persona="Formal Jarvis",
            memory_ttl_hours=12,
            opt_in_privacy=True,
            preferred_tools=["notes", "weather"],
        )
        self.assertIsInstance(prof, UserProfile)
        self.assertEqual(prof.user_id, "user_alpha")
        self.assertEqual(prof.persona, "Formal Jarvis")
        self.assertEqual(prof.memory_ttl_hours, 12)
        self.assertTrue(prof.opt_in_privacy)
        self.assertEqual(prof.preferred_tools, ["notes", "weather"])

        # Fetch back
        fetched = self.mgr.get_profile("user_alpha")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.persona, "Formal Jarvis")

    def test_get_nonexistent_profile(self):
        res = self.mgr.get_profile("unknown_user")
        self.assertIsNone(res)

    def test_delete_profile(self):
        self.mgr.set_profile(user_id="user_del", persona="Casual")
        self.assertIsNotNone(self.mgr.get_profile("user_del"))
        self.mgr.delete_profile("user_del")
        self.assertIsNone(self.mgr.get_profile("user_del"))


class TestConversationManager(unittest.TestCase):
    def setUp(self):
        self.cm = ConversationManager(max_turns=5)

    def test_add_turns_and_rolling_window(self):
        for i in range(7):
            self.cm.add_turn("user" if i % 2 == 0 else "assistant", f"Message {i}")

        self.assertLessEqual(len(self.cm._history), 5)
        self.assertEqual(self.cm._history[-1].content, "Message 6")

    def test_build_messages(self):
        self.cm.add_turn("user", "Hello computer")
        self.cm.add_turn("assistant", "Hello human")
        msgs = self.cm.build_messages("What is my name?")
        self.assertTrue(any("Hello computer" in m["content"] for m in msgs))
        self.assertEqual(msgs[-1]["content"], "What is my name?")

    def test_get_recent_context_and_clear(self):
        self.cm.add_turn("user", "Testing context summary")
        summary = self.cm.get_recent_context(n=1)
        self.assertIn("Testing context summary", summary)
        self.cm.clear_session()
        self.assertEqual(len(self.cm._history), 0)


class TestLoggerAndFormatters(unittest.TestCase):
    def test_json_formatter(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="TestLogger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Structured test event",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["message"], "Structured test event")
        self.assertEqual(data["logger"], "TestLogger")

    def test_console_formatter(self):
        formatter = ConsoleFormatter()
        record = logging.LogRecord(
            name="TestLogger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Console error output",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        self.assertIn("Console error output", formatted)

    def test_setup_logging(self):
        setup_logging(level=logging.DEBUG)
        log = logging.getLogger("SentinelUnitTest")
        self.assertIsNotNone(log)


class TestTypedExceptions(unittest.TestCase):
    def test_exception_hierarchy_and_formatting(self):
        err = SentinelError("Core failure", module="AudioEngine", original_error=ValueError("Invalid sample rate"))
        msg = str(err)
        self.assertIn("[AudioEngine]", msg)
        self.assertIn("Core failure", msg)
        self.assertIn("Invalid sample rate", msg)

        # Derived classes
        self.assertIsInstance(HardwareError("Mic unplugged"), SentinelError)
        self.assertIsInstance(LLMServiceError("Model timeout"), SentinelError)
        self.assertIsInstance(VaultError("Decryption failed"), SentinelError)
        self.assertIsInstance(ActionExecutionError("Click failed"), SentinelError)
        self.assertIsInstance(SandboxError("Forbidden import"), SentinelError)


class TestAppRegistryFunctions(unittest.TestCase):
    def test_normalize_name(self):
        self.assertEqual(normalize_name("  Google Chrome  "), "google chrome")
        self.assertEqual(normalize_name(""), "")
        self.assertEqual(normalize_name(None), "")

    def test_save_and_load_apps_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = os.path.join(tmpdir, "apps.json")
            with patch("sentinel.app.registry.APPS_REGISTRY_PATH", reg_path):
                save_apps_index({"notepad": "notepad.exe"})
                loaded = load_apps_index()
                self.assertEqual(loaded.get("notepad"), "notepad.exe")

    def test_save_and_load_mru(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mru_path = os.path.join(tmpdir, "mru.json")
            with patch("sentinel.app.registry.MRU_PATH", mru_path):
                save_mru({"calculator": 5})
                loaded = load_mru()
                self.assertEqual(loaded.get("calculator"), 5)


class TestSecretsCryptoCore(unittest.TestCase):
    def test_pbkdf_and_keystream(self):
        salt = os.urandom(16)
        key = _pbkdf("SecretPassphrase", salt, length=32, rounds=1000)
        self.assertEqual(len(key), 32)

        nonce = os.urandom(12)
        stream = _keystream(key, nonce, length=64)
        self.assertEqual(len(stream), 64)

    @unittest.skipUnless(AESGCM is not None, "cryptography package required for AES-GCM vaulting")
    def test_encrypt_decrypt_secret(self):
        original = "SUPER_SECRET_API_KEY_12345"
        salt, nonce, ciphertext = encrypt_data("MasterPassword", original)
        self.assertTrue(len(ciphertext) > 0)
        recovered = decrypt_data("MasterPassword", salt, nonce, ciphertext)
        self.assertEqual(recovered, original)

    def test_master_password_hash_and_verify(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, "vault.db")
            with patch("sentinel.app.secrets.SECRETS_DB", db_file):
                self.assertFalse(check_master_exists())
                set_master_password("AlphaMaster123")
                self.assertTrue(check_master_exists())
                self.assertTrue(verify_master("AlphaMaster123"))
                self.assertFalse(verify_master("WrongPassword"))

    @unittest.skipUnless(AESGCM is not None, "cryptography package required for AES-GCM vaulting")
    def test_master_password_vault(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, "vault.db")
            with patch("sentinel.app.secrets.SECRETS_DB", db_file):
                set_master_password("AlphaMaster123")
                store_secret("TEST_KEY", "VAL123", "AlphaMaster123")
                self.assertEqual(fetch_secret("TEST_KEY", "AlphaMaster123"), "VAL123")


class TestCodingModule(unittest.TestCase):
    def setUp(self):
        self.coder = CodingModule()

    def test_validate_code_valid(self):
        valid, err = self.coder.validate_code("def add(a, b):\n    return a + b\n")
        self.assertTrue(valid)
        self.assertEqual(err, "")

    def test_validate_code_syntax_error(self):
        valid, err = self.coder.validate_code("def broken(:\n    pass\n")
        self.assertFalse(valid)
        self.assertIn("invalid syntax", err.lower())

    def test_extract_code(self):
        snippet = "Here is the code:\n```python\nprint('hello')\n```\nEnjoy!"
        extracted = self.coder._extract_code(snippet)
        self.assertIn("print('hello')", extracted)

    def test_execute_code_sandbox_safe(self):
        output = self.coder.execute_python_code("print('Sentinel Coding Test')")
        self.assertIn("Sentinel Coding Test", output)


class TestSystemMonitor(unittest.TestCase):
    def setUp(self):
        self.mon = SystemMonitor(temp_threshold=70)

    def test_check_temp_alerts_normal(self):
        with patch.object(self.mon, "get_cpu_temp", return_value=45.0):
            with patch.object(self.mon, "get_gpu_temp", return_value=50.0):
                alerts = self.mon.check_temp_alerts()
                self.assertEqual(len(alerts), 0)

    def test_check_temp_alerts_high(self):
        with patch.object(self.mon, "get_cpu_temp", return_value=85.0):
            with patch.object(self.mon, "get_gpu_temp", return_value=82.0):
                alerts = self.mon.check_temp_alerts()
                self.assertEqual(len(alerts), 2)
                self.assertIn("CPU temperature high", alerts[0])

    def test_check_misuse(self):
        alerts = self.mon.check_misuse()
        self.assertIsInstance(alerts, list)


class TestHealthMonitor(unittest.TestCase):
    def test_health_monitor_metrics(self):
        hm = HealthMonitor(interval=60)
        hm._update_system_metrics()
        hm.record_latency("gemini", 120.0)
        hm.record_latency("gemini", 3500.0)
        self.assertIsNotNone(hm.state)


class TestIntelligenceModule(unittest.TestCase):
    def test_search_the_web_empty(self):
        res = search_the_web("")
        self.assertIn("No query", res)

    def test_gemini_generate_fallback(self):
        # Without valid GEMINI_API_KEY, should trigger safe fallback
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            res = gemini_generate("Hello test")
            self.assertIsInstance(res, str)


class TestVoiceHelpers(unittest.TestCase):
    def test_fast_voice(self):
        fv = FastVoiceHandler(speak_fn=lambda t, **k: None)
        result = fv.execute_with_announcement("quick task", lambda: "Executed result", pre_speak=None, post_speak=None)
        self.assertEqual(result, "Executed result")

    def test_enhanced_tts_fallback(self):
        tts = EnhancedTTS()
        with patch.object(tts, "_speak_fallback", return_value=True):
            with patch.object(tts, "_speak_edge", return_value=True):
                res = tts.speak("Test speech phrase")
                self.assertTrue(res)


class TestCalendarModule(unittest.TestCase):
    def test_calendar_no_creds(self):
        cal = CalendarModule(credentials_path="nonexistent_creds.json")
        ok, msg = cal.authenticate()
        self.assertFalse(ok)
        self.assertIn("missing", msg)


class TestSentinelOrchestratorSuite(unittest.TestCase):
    def setUp(self):
        with patch.object(SentinelOrchestrator, "_start_background_services"):
            self.orch = SentinelOrchestrator(
                llm_callback=lambda p: "Orchestrator LLM Output",
                speak_fn=lambda t, **kw: None,
            )

    def test_orchestrator_initialization_and_attributes(self):
        self.assertIsNotNone(self.orch.system_monitor)
        self.assertIsNotNone(self.orch.coding_module)
        self.assertIsNotNone(self.orch.calendar_module)
        self.assertIsNotNone(self.orch.command_router)

    def test_safe_llm_call(self):
        res = self.orch._safe_llm_call("test prompt")
        self.assertEqual(res, "Orchestrator LLM Output")

    def test_run_command_dispatch(self):
        out = self.orch.run_command("what is 5 + 5")
        self.assertIsNotNone(out)


if __name__ == "__main__":
    unittest.main()
