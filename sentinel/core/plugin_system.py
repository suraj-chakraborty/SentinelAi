"""
sentinel/core/plugin_system.py
──────────────────────────────
Manifest-based Plugin System — Tier 3 Differentiator.

Allows community/user-built plugins to extend SentinelAI without
modifying core code. Each plugin is a folder with:

    my_plugin/
    ├── manifest.json    ← metadata, intents, trigger phrases
    └── plugin.py        ← Python module with a Plugin class

manifest.json schema:
{
    "name": "Spotify Controller",
    "version": "1.0.0",
    "description": "Control Spotify playback via voice",
    "author": "SentinelAI Community",
    "intents": ["play_music", "pause_music", "next_track"],
    "triggers": ["play", "pause music", "next song", "skip track"],
    "entry": "plugin.py",
    "class": "SpotifyPlugin",
    "requires": ["spotipy>=2.23"]
}

plugin.py must expose a class with:
    class MyPlugin:
        def __init__(self, orchestrator): ...
        def can_handle(self, command: str) -> bool: ...
        def handle(self, command: str) -> str: ...
        def on_load(self): ...
        def on_unload(self): ...
"""

import os
import sys
import json
import importlib.util
import logging
import subprocess
from typing import Dict, List, Optional

logger = logging.getLogger("PluginSystem")

PLUGINS_DIR = os.path.join(
    os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi", "plugins"
)
BUILTIN_PLUGINS_DIR = os.path.join(os.path.dirname(__file__), "..", "plugins")


class PluginBase:
    """Base class all plugins should inherit from."""

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(f"Plugin.{self.__class__.__name__}")

    def can_handle(self, command: str) -> bool:
        """Return True if this plugin can handle the given command."""
        return False

    def handle(self, command: str) -> str:
        """Handle the command and return a response string."""
        return ""

    def on_load(self):
        """Called when the plugin is loaded."""
        pass

    def on_unload(self):
        """Called when the plugin is unloaded."""
        pass


class LoadedPlugin:
    """Wraps a loaded plugin with its manifest."""

    def __init__(self, manifest: dict, instance: PluginBase, path: str):
        self.manifest = manifest
        self.instance = instance
        self.path = path
        self.name = manifest.get("name", "Unknown")
        self.version = manifest.get("version", "0.0")
        self.triggers: List[str] = [t.lower() for t in manifest.get("triggers", [])]

    def matches(self, command: str) -> bool:
        """Check if command matches any trigger phrase, or plugin.can_handle()."""
        cmd_lower = command.lower()
        if any(trigger in cmd_lower for trigger in self.triggers):
            return True
        try:
            return self.instance.can_handle(command)
        except Exception:
            return False

    def execute(self, command: str) -> str:
        """Execute the plugin handler."""
        try:
            return self.instance.handle(command) or ""
        except Exception as e:
            self.instance.logger.error("Plugin '%s' handle error: %s", self.name, e, exc_info=True)
            return f"Plugin error: {e}"


class PluginSystem:
    """
    Discovers, loads, and dispatches to plugins.

    Usage:
        ps = PluginSystem(orchestrator=orchestrator)
        ps.load_all()

        # In command router:
        result = ps.dispatch(command)
        if result is not None:
            speak(result)
    """

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._plugins: Dict[str, LoadedPlugin] = {}
        os.makedirs(PLUGINS_DIR, exist_ok=True)

    # ─── Discovery & Loading ──────────────────────────────────────────────────

    def load_all(self) -> int:
        """Scan plugin directories and load all valid plugins. Returns count loaded."""
        loaded = 0
        for plugins_dir in [BUILTIN_PLUGINS_DIR, PLUGINS_DIR]:
            if not os.path.isdir(plugins_dir):
                continue
            for entry in os.scandir(plugins_dir):
                if entry.is_dir():
                    manifest_path = os.path.join(entry.path, "manifest.json")
                    if os.path.exists(manifest_path):
                        if self.load_plugin(entry.path):
                            loaded += 1
        logger.info(f"Loaded {loaded} plugin(s).")
        return loaded

    def load_plugin(self, plugin_dir: str) -> bool:
        """Load a single plugin from its directory, installing any missing dependencies first."""
        manifest_path = os.path.join(plugin_dir, "manifest.json")
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            logger.error("Bad manifest at %s: %s", manifest_path, exc)
            return False

        name = manifest.get("name", os.path.basename(plugin_dir))

        # ── Install dependencies declared in manifest ──────────────────────
        requires = manifest.get("requires", [])
        if requires:
            if not self._install_requirements(name, requires):
                logger.warning("Plugin '%s' skipped — dependency installation failed.", name)
                return False

        entry_file = manifest.get("entry", "plugin.py")
        class_name = manifest.get("class", "Plugin")
        entry_path = os.path.join(plugin_dir, entry_file)

        if not os.path.exists(entry_path):
            logger.error("Plugin '%s': entry file not found: %s", name, entry_path)
            return False

        try:
            spec = importlib.util.spec_from_file_location(f"sentinel_plugin_{name}", entry_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            PluginClass = getattr(module, class_name)
            instance = PluginClass(orchestrator=self.orchestrator)
            instance.on_load()
            self._plugins[name] = LoadedPlugin(manifest, instance, plugin_dir)
            logger.info("Loaded plugin: %s v%s", name, manifest.get("version", "?"))
            return True
        except Exception as exc:
            logger.error("Failed to load plugin '%s': %s", name, exc, exc_info=True)
            return False

    def _install_requirements(self, plugin_name: str, requires: List[str]) -> bool:
        """
        Install each requirement in `requires` via pip if not already present.
        Returns True if all requirements are satisfied, False if any failed.
        """
        all_ok = True
        for req in requires:
            pkg = req.split(">=")[0].split("<=")[0].split("==")[0].split("!=")[0].strip()
            try:
                __import__(pkg.replace("-", "_"))
                logger.debug("Plugin '%s' dep already satisfied: %s", plugin_name, pkg)
            except ImportError:
                logger.info("Installing dependency '%s' for plugin '%s'…", req, plugin_name)
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", req, "--quiet"],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    if result.returncode == 0:
                        logger.info("Installed: %s", req)
                    else:
                        logger.error("pip install failed for '%s': %s", req, result.stderr[:200])
                        all_ok = False
                except Exception as exc:
                    logger.error("Dependency install error for '%s': %s", req, exc)
                    all_ok = False
        return all_ok

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin by name."""
        if name in self._plugins:
            try:
                self._plugins[name].instance.on_unload()
            except Exception:
                pass
            del self._plugins[name]
            logger.info(f"Unloaded plugin: {name}")
            return True
        return False

    def reload_plugin(self, name: str) -> bool:
        """Reload a plugin in-place."""
        path = self._plugins.get(name, LoadedPlugin.__new__(LoadedPlugin)).path if name in self._plugins else None
        self.unload_plugin(name)
        if path:
            return self.load_plugin(path)
        return False

    # ─── Dispatch ─────────────────────────────────────────────────────────────

    def dispatch(self, command: str) -> Optional[str]:
        """
        Try each plugin in order. Return the first non-empty response,
        or None if no plugin handles the command.
        """
        for name, plugin in self._plugins.items():
            if plugin.matches(command):
                logger.info(f"Plugin '{name}' handling: {command[:40]}")
                result = plugin.execute(command)
                if result:
                    return result
        return None

    # ─── Introspection ────────────────────────────────────────────────────────

    def list_plugins(self) -> List[dict]:
        """Returns info about all loaded plugins."""
        return [
            {
                "name": p.name,
                "version": p.version,
                "triggers": p.triggers,
                "path": p.path
            }
            for p in self._plugins.values()
        ]

    def is_loaded(self, name: str) -> bool:
        return name in self._plugins

    def __len__(self):
        return len(self._plugins)
