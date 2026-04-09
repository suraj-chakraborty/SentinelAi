"""
sentinel/core/dynamic_plugin_loader.py
──────────────────────────────────────
Dynamic plugin loader that can load plugins at runtime
including auto-generated plugins.
"""

import os
import logging
from typing import Optional, List, Dict, Any

from sentinel.core.plugin_system import PluginSystem, LoadedPlugin, PLUGINS_DIR
from sentinel.core.plugin_generator import GENERATED_PLUGINS_DIR

logger = logging.getLogger("DynamicPluginLoader")


class DynamicPluginLoader:
    """Extends PluginSystem with runtime plugin loading capabilities."""

    def __init__(self, plugin_system: PluginSystem):
        self._plugin_system = plugin_system
        self._loaded_paths: set = set()

    def load_generated_plugins(self) -> int:
        """Load all auto-generated plugins."""
        if not os.path.exists(GENERATED_PLUGINS_DIR):
            return 0
        
        loaded = 0
        for entry in os.scandir(GENERATED_PLUGINS_DIR):
            if entry.is_dir():
                manifest_path = os.path.join(entry.path, "manifest.json")
                if os.path.exists(manifest_path) and entry.path not in self._loaded_paths:
                    if self._plugin_system.load_plugin(entry.path):
                        self._loaded_paths.add(entry.path)
                        loaded += 1
        
        logger.info(f"Loaded {loaded} generated plugin(s)")
        return loaded

    def load_plugin_at_path(self, plugin_dir: str) -> bool:
        """Load a plugin from a specific directory path."""
        if plugin_dir in self._loaded_paths:
            logger.debug(f"Plugin already loaded: {plugin_dir}")
            return True
        
        if self._plugin_system.load_plugin(plugin_dir):
            self._loaded_paths.add(plugin_dir)
            return True
        return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin by name."""
        if plugin_name in self._plugin_system._plugins:
            plugin = self._plugin_system._plugins[plugin_name]
            try:
                plugin.instance.on_unload()
            except Exception as e:
                logger.warning(f"Plugin on_unload error: {e}")
            
            del self._plugin_system._plugins[plugin_name]
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
        return False

    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin (unload then load)."""
        plugin_path = None
        if plugin_name in self._plugin_system._plugins:
            plugin_path = self._plugin_system._plugins[plugin_name].path
            self.unload_plugin(plugin_name)
        
        if plugin_path:
            return self.load_plugin_at_path(plugin_path)
        return False

    def get_loaded_plugins(self) -> List[Dict[str, Any]]:
        """Get list of all loaded plugins."""
        plugins = []
        for name, plugin in self._plugin_system._plugins.items():
            plugins.append({
                "name": name,
                "version": plugin.version,
                "path": plugin.path,
                "triggers": plugin.triggers
            })
        return plugins

    def discover_unloaded_plugins(self) -> List[Dict[str, str]]:
        """Find plugins in plugin directories that aren't loaded."""
        unloaded = []
        for plugins_dir in [GENERATED_PLUGINS_DIR]:
            if not os.path.isdir(plugins_dir):
                continue
            for entry in os.scandir(plugins_dir):
                if entry.is_dir() and entry.path not in self._loaded_paths:
                    manifest_path = os.path.join(entry.path, "manifest.json")
                    if os.path.exists(manifest_path):
                        unloaded.append({
                            "name": entry.name,
                            "path": entry.path
                        })
        return unloaded


def create_dynamic_loader(plugin_system: PluginSystem) -> DynamicPluginLoader:
    """Factory function to create a dynamic plugin loader."""
    return DynamicPluginLoader(plugin_system)