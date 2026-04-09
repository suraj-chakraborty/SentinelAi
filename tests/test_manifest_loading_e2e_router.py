import json
import os
import tempfile
import unittest
from unittest import mock


class TestManifestLoadingE2ERouter(unittest.TestCase):
    def test_end_to_end_dispatch_through_router(self):
        # Setup a temporary manifest plugin and patch the PLUGINS_DIR
        from sentinel.core import command_router as cr
        import sentinel.core.command_router as cr_mod
        from sentinel.core.command_router import CommandRouter
        from sentinel.core.plugin_system import PluginSystem

        with tempfile.TemporaryDirectory() as tmpdir:
            plugins_root = os.path.join(tmpdir, "plugins_e2e")
            os.makedirs(os.path.join(plugins_root, "ManifestEndToEnd"), exist_ok=True)
            manifest = {
                "name": "ManifestEndToEnd",
                "version": "0.1.0",
                "description": "End-to-end manifest test plugin",
                "author": "Test",
                "intents": [],
                "triggers": ["test manifest plugin"],
                "entry": "plugin.py",
                "class": "TestManifestPlugin",
                "requires": [],
            }
            with open(os.path.join(plugins_root, "ManifestEndToEnd", "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with open(os.path.join(plugins_root, "ManifestEndToEnd", "plugin.py"), "w", encoding="utf-8") as f:
                f.write(
                    """
from sentinel.core.plugin_system import PluginBase
class TestManifestPlugin(PluginBase):
    def __init__(self, orchestrator=None):
        super().__init__(orchestrator)
    def can_handle(self, command):
        return True
    def execute(self, command, entity):
        return 'Manifest Plugin Activated'
"""
                )

            import sentinel.core.plugin_system as plugin_sys
            import sentinel.core.plugin_system as plugin_sys
            old_dir = plugin_sys.PLUGINS_DIR
            try:
                plugin_sys.PLUGINS_DIR = plugins_root
                class DummyOrchestrator: pass
                router = CommandRouter(orchestrator=DummyOrchestrator())
                result = router.run_command("test manifest plugin")
                self.assertEqual(result, "Manifest Plugin Activated")
            finally:
                plugin_sys.PLUGINS_DIR = old_dir


if __name__ == '__main__':
    unittest.main()
