import json
import os
import tempfile
import unittest


class TestManifestLoadingEdgecases(unittest.TestCase):
    def test_missing_manifest_ignored(self):
        # Ensure a directory with no manifest.json doesn't crash the loader
        from sentinel.core.plugin_system import PluginSystem
        import sentinel.core.plugin_system as ps_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            plugins_root = os.path.join(tmpdir, "plugins_empty")
            os.makedirs(plugins_root, exist_ok=True)
            # Subdir without manifest.json
            os.makedirs(os.path.join(plugins_root, "NoManifestPlugin"), exist_ok=True)

            import sentinel.core.plugin_system as plugin_sys
            old_dir = getattr(plugin_sys, 'PLUGINS_DIR', None)
            old_builtin = getattr(plugin_sys, 'BUILTIN_PLUGINS_DIR', None)
            try:
                plugin_sys.PLUGINS_DIR = plugins_root
                if old_builtin is not None:
                    plugin_sys.BUILTIN_PLUGINS_DIR = os.path.join(tmpdir, "builtin_empty_edge_missing1")
                    os.makedirs(plugin_sys.BUILTIN_PLUGINS_DIR, exist_ok=True)
                class DummyOrchestrator: pass
                ps = PluginSystem(orchestrator=DummyOrchestrator())
                loaded = ps.load_all()
                self.assertEqual(loaded, 0)
            finally:
                if old_dir is not None:
                    plugin_sys.PLUGINS_DIR = old_dir
                if old_builtin is not None:
                    plugin_sys.BUILTIN_PLUGINS_DIR = old_builtin

    def test_missing_entry_ignored(self):
        # manifest.json refers to a non-existent entry file
        from sentinel.core.plugin_system import PluginSystem
        import sentinel.core.plugin_system as ps_mod
        with tempfile.TemporaryDirectory() as tmpdir:
            plugins_root = os.path.join(tmpdir, "plugins_missing_entry")
            os.makedirs(os.path.join(plugins_root, "MissingEntry"), exist_ok=True)
            manifest = {
                "name": "MissingEntry",
                "version": "0.1.0",
                "description": "Plugin with missing entry",
                "author": "Test",
                "intents": [],
                "triggers": ["missing entry"],
                "entry": "nonexistent.py",
                "class": "MissingEntry",
                "requires": [],
            }
            with open(os.path.join(plugins_root, "MissingEntry", "manifest.json"), "w", encoding="utf-8") as f:
                import json
                json.dump(manifest, f)
            import sentinel.core.plugin_system as plugin_sys
            old_dir = getattr(plugin_sys, 'PLUGINS_DIR', None)
            old_builtin = getattr(plugin_sys, 'BUILTIN_PLUGINS_DIR', None)
            try:
                plugin_sys.PLUGINS_DIR = plugins_root
                if old_builtin is not None:
                    plugin_sys.BUILTIN_PLUGINS_DIR = os.path.join(tempfile.gettempdir(), "builtin_empty_edge_missing2")
                    os.makedirs(plugin_sys.BUILTIN_PLUGINS_DIR, exist_ok=True)
                class DummyOrchestrator: pass
                ps = PluginSystem(orchestrator=DummyOrchestrator())
                loaded = ps.load_all()
                self.assertEqual(loaded, 0)
            finally:
                plugin_sys.PLUGINS_DIR = old_dir
                if old_builtin is not None:
                    plugin_sys.BUILTIN_PLUGINS_DIR = old_builtin
                

    def test_missing_class_ignored(self):
        # manifest points to plugin.py, but class name doesn't exist in plugin.py
        from sentinel.core.plugin_system import PluginSystem
        import sentinel.core.plugin_system as ps_mod
        with tempfile.TemporaryDirectory() as tmpdir:
            plugins_root = os.path.join(tmpdir, "plugins_missing_class")
            os.makedirs(os.path.join(plugins_root, "MissingClass"), exist_ok=True)
            manifest = {
                "name": "MissingClass",
                "version": "0.1.0",
                "description": "Plugin with missing class",
                "author": "Test",
                "intents": [],
                "triggers": ["missing class"],
                "entry": "plugin.py",
                "class": "NonExistentClass",
                "requires": [],
            }
            with open(os.path.join(plugins_root, "MissingClass", "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with open(os.path.join(plugins_root, "MissingClass", "plugin.py"), "w", encoding="utf-8") as f:
                f.write("class SomeOtherName:\n    pass\n")
            import sentinel.core.plugin_system as plugin_sys
            old_dir = plugin_sys.PLUGINS_DIR
            old_builtin = getattr(plugin_sys, 'BUILTIN_PLUGINS_DIR', None)
            try:
                plugin_sys.PLUGINS_DIR = plugins_root
                if old_builtin is not None:
                    plugin_sys.BUILTIN_PLUGINS_DIR = os.path.join(tempfile.gettempdir(), "builtin_empty_edge2")
                    os.makedirs(plugin_sys.BUILTIN_PLUGINS_DIR, exist_ok=True)
                class DummyOrchestrator: pass
                ps = PluginSystem(orchestrator=DummyOrchestrator())
                loaded = ps.load_all()
                self.assertEqual(loaded, 0)
            finally:
                plugin_sys.PLUGINS_DIR = old_dir
                if old_builtin is not None:
                    plugin_sys.BUILTIN_PLUGINS_DIR = old_builtin

    def test_dependency_install_failure(self):
        # Simulate dependency install failure via patching _install_requirements
        from sentinel.core.plugin_system import PluginSystem
        import sentinel.core.plugin_system as ps_mod
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmpdir:
            plugins_root = os.path.join(tmpdir, "plugins_dep_fail")
            os.makedirs(os.path.join(plugins_root, "DepFail"), exist_ok=True)
            manifest = {
                "name": "DepFail",
                "version": "0.1.0",
                "description": "Plugin with failing dependencies",
                "author": "Test",
                "intents": [],
                "triggers": ["dep fail"],
                "entry": "plugin.py",
                "class": "DepFail",
                "requires": ["nonexistentpkg>=0.0.0"],
            }
            with open(os.path.join(plugins_root, "DepFail", "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with open(os.path.join(plugins_root, "DepFail", "plugin.py"), "w", encoding="utf-8") as f:
                f.write("from sentinel.core.plugin_system import PluginBase\nclass DepFail(PluginBase):\n    def __init__(self, orchestrator=None): super().__init__(orchestrator)\n    def can_handle(self, c): return False\n    def execute(self, c, e): return 'dep ok'\n")
            old_dir = ps_mod.PLUGINS_DIR
            try:
                ps_mod.PLUGINS_DIR = plugins_root
                class DummyOrchestrator: pass
                ps = PluginSystem(orchestrator=DummyOrchestrator())
                with mock.patch.object(PluginSystem, "_install_requirements", return_value=False):
                    loaded = ps.load_all()
                    self.assertEqual(loaded, 0)
            finally:
                ps_mod.PLUGINS_DIR = old_dir


if __name__ == '__main__':
    unittest.main()
