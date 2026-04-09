import json
import os
import tempfile
import unittest


class TestManifestLoading(unittest.TestCase):
    def test_manifest_loading_with_mock_plugin(self):
        # Import after creating temp plugin path to avoid polluting global state
        from sentinel.core.plugin_system import PluginSystem
        import sentinel.core.plugin_system as ps_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            plugins_dir = os.path.join(tmpdir, "plugins")
            os.makedirs(os.path.join(plugins_dir, "TestManifest"), exist_ok=True)

            manifest = {
                "name": "TestManifest",
                "version": "0.1.0",
                "description": "Test manifest-based plugin",
                "author": "Test",
                "intents": [],
                "triggers": ["test manifest plugin"],
                "entry": "plugin.py",
                "class": "TestManifest",
                "requires": [],
            }

            with open(os.path.join(plugins_dir, "TestManifest", "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f)

            plugin_py = os.path.join(plugins_dir, "TestManifest", "plugin.py")
            with open(plugin_py, "w", encoding="utf-8") as f:
                f.write(
                    """
from sentinel.core.plugin_system import PluginBase
class TestManifest(PluginBase):
    def __init__(self, orchestrator=None):
        super().__init__(orchestrator)
    def can_handle(self, command):
        return True
    def execute(self, command, entity):
        return "Manifest Plugin Activated"
"""
                )

            # Patch the PLUGINS_DIR used by the PluginSystem to our temp dir
            old_dir = ps_mod.PLUGINS_DIR
            ps_mod.PLUGINS_DIR = plugins_dir
            try:
                class DummyOrchestrator: pass
                ps = PluginSystem(orchestrator=DummyOrchestrator())
                loaded = ps.load_all()
                # There is at least one plugin loaded (the builtins), but our manifest should be discovered too
            self.assertGreaterEqual(loaded, 1)
            self.assertIn("TestManifest", ps._plugins)
                res = ps.dispatch("test manifest plugin")
                self.assertEqual(res, "Manifest Plugin Activated")
            finally:
                ps_mod.PLUGINS_DIR = old_dir


if __name__ == '__main__':
    unittest.main()
