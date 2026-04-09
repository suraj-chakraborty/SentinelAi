"""
tests/test_plugin_generator.py
───────────────────────────────
Tests for self-healing plugin generation.
"""

import unittest
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel.core.plugin_generator import (
    PluginGenerator, sanitize_name, GENERATED_PLUGINS_DIR
)


class TestSanitizeName(unittest.TestCase):
    def test_sanitize_valid(self):
        self.assertEqual(sanitize_name("my_plugin"), "my_plugin")
    
    def test_sanitize_spaces(self):
        self.assertEqual(sanitize_name("my plugin name"), "my_plugin_name")
    
    def test_sanitize_special_chars(self):
        self.assertEqual(sanitize_name("test@plugin#1"), "test_plugin_1")


class TestPluginGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = GENERATED_PLUGINS_DIR
        import sentinel.core.plugin_generator as pg
        pg.GENERATED_PLUGINS_DIR = self.temp_dir
        self.gen = PluginGenerator(orchestrator=None)

    def tearDown(self):
        import sentinel.core.plugin_generator as pg
        pg.GENERATED_PLUGINS_DIR = self.original_dir
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_without_orchestrator(self):
        success, name, msg = self.gen.generate_plugin("test command")
        self.assertFalse(success)
        self.assertIn("Orchestrator", msg)

    def test_analyze_command_fallback(self):
        name, triggers, intents = self.gen._analyze_command("test command", None)
        self.assertIsNotNone(name)
        self.assertIsInstance(triggers, list)
        self.assertIsInstance(intents, list)

    def test_list_generated_plugins_empty(self):
        plugins = self.gen.list_generated_plugins()
        self.assertEqual(plugins, [])


if __name__ == "__main__":
    unittest.main()