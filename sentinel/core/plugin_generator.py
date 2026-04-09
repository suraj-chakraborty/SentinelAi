"""
sentinel/core/plugin_generator.py
──────────────────────────────────
Self-healing plugin generation — auto-creates plugins for unknown commands.

When the command router encounters a command it cannot handle,
this module generates a new manifest-based plugin using LLM.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from sentinel.core.plugin_system import PLUGINS_DIR
from sentinel.app.config import APPDATA_DIR

logger = logging.getLogger("PluginGenerator")

GENERATED_PLUGINS_DIR = os.path.join(APPDATA_DIR, "plugins", "generated")
Path(GENERATED_PLUGINS_DIR).mkdir(parents=True, exist_ok=True)

PLUGIN_TEMPLATE = '''"""
sentinel/plugins/{plugin_name}/plugin.py
Auto-generated plugin for: {command_description}
"""

import logging
from sentinel.core.plugin_system import PluginBase


class {class_name}(PluginBase):
    """Auto-generated plugin for handling: {trigger_description}"""

    def __init__(self, orchestrator=None):
        super().__init__(orchestrator)
        self.name = "{plugin_name}"
        self.triggers = {triggers}

    def can_handle(self, command: str) -> bool:
        """Check if command matches our triggers."""
        cmd_lower = command.lower()
        return any(t.lower() in cmd_lower for t in self.triggers)

    def handle(self, command: str) -> str:
        """Handle the command."""
        # TODO: Implement custom logic for this command
        # Access orchestrator: self.orchestrator
        # Return response string
        
        action = self._extract_action(command)
        
        if action == "execute":
            # Example: perform the action
            pass
        
        return "Executing: {command_description}"

    def _extract_action(self, command: str) -> str:
        """Extract the action to perform from the command."""
        # Simple keyword-based extraction - can be enhanced
        cmd_lower = command.lower()
        if "create" in cmd_lower or "make" in cmd_lower:
            return "create"
        elif "delete" in cmd_lower or "remove" in cmd_lower:
            return "delete"
        elif "show" in cmd_lower or "list" in cmd_lower:
            return "list"
        elif "get" in cmd_lower or "fetch" in cmd_lower:
            return "get"
        return "execute"

    def health(self) -> bool:
        """Health check - default to healthy."""
        return True

    def on_load(self):
        """Called when plugin loads."""
        logger.info(f"Plugin '{self.name}' loaded")

    def on_unload(self):
        """Called when plugin unloads."""
        logger.info(f"Plugin '{self.name}' unloaded")
'''

MANIFEST_TEMPLATE = '''{{
    "name": "{plugin_name}",
    "version": "1.0.0",
    "description": "Auto-generated plugin for: {description}",
    "author": "SentinelAI Auto-Generator",
    "intents": [{intents}],
    "triggers": {triggers},
    "entry": "plugin.py",
    "class": "{class_name}",
    "requires": []
}}
'''


def sanitize_name(name: str) -> str:
    """Sanitize plugin name for filesystem and Python identifiers."""
    keep = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    return keep.strip("_").lower()


class PluginGenerator:
    """Generates plugins for unknown commands using LLM."""

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    def generate_plugin(
        self,
        command: str,
        description: Optional[str] = None
    ) -> Tuple[bool, str, str]:
        """
        Generate a plugin for the given command.
        
        Returns: (success, plugin_name, message)
        """
        if not self.orchestrator:
            return False, "", "Orchestrator not available for LLM generation"

        try:
            plugin_name, triggers, intents = self._analyze_command(command, description)
            
            if not plugin_name:
                return False, "", "Could not analyze command"

            plugin_name = sanitize_name(plugin_name)
            class_name = "".join(word.title() for word in plugin_name.replace("-", "_").split("_"))
            
            plugin_dir = os.path.join(GENERATED_PLUGINS_DIR, plugin_name)
            Path(plugin_dir).mkdir(parents=True, exist_ok=True)

            self._write_plugin_file(plugin_dir, plugin_name, class_name, command, triggers)
            self._write_manifest(plugin_dir, plugin_name, class_name, command, triggers, intents)

            logger.info(f"Generated plugin: {plugin_name} at {plugin_dir}")
            return True, plugin_name, f"Plugin '{plugin_name}' created successfully"

        except Exception as e:
            logger.error(f"Plugin generation failed: {e}")
            return False, "", str(e)

    def _analyze_command(
        self,
        command: str,
        description: Optional[str] = None
    ) -> Tuple[Optional[str], list, list]:
        """Use LLM to analyze command and extract plugin metadata."""
        prompt = f"""Analyze this voice command and extract plugin metadata.
Command: {command}
Description: {description or 'N/A'}

Output a JSON object with:
- name: short plugin name (lowercase, underscores)
- triggers: list of trigger phrases users might say
- intents: list of intents this command fulfills

Only respond with valid JSON, no other text.
Example: {{"name": "weather_check", "triggers": ["weather", "forecast"], "intents": ["get_weather"]}}"""

        try:
            response = self.orchestrator._safe_llm_call(prompt)
            
            for fence in ("```json", "```"):
                if fence in response:
                    response = response.split(fence)[1].split("```")[0].strip()
                    break
            
            data = json.loads(response)
            name = data.get("name", "custom_plugin")
            triggers = data.get("triggers", [command[:30]])
            intents = data.get("intents", ["custom"])
            
            return name, triggers, intents

        except json.JSONDecodeError:
            fallback_name = sanitize_name(command.split()[0] if command else "custom")
            return fallback_name, [command[:50]], ["custom"]
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            fallback_name = sanitize_name(command.split()[0] if command else "custom")
            return fallback_name, [command[:50]], ["custom"]

    def _write_plugin_file(
        self,
        plugin_dir: str,
        plugin_name: str,
        class_name: str,
        command: str,
        triggers: list
    ):
        content = PLUGIN_TEMPLATE.format(
            plugin_name=plugin_name,
            class_name=class_name,
            command_description=command,
            trigger_description=", ".join(triggers[:3]),
            triggers=triggers
        )
        
        filepath = os.path.join(plugin_dir, "plugin.py")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def _write_manifest(
        self,
        plugin_dir: str,
        plugin_name: str,
        class_name: str,
        command: str,
        triggers: list,
        intents: list
    ):
        triggers_json = json.dumps([t for t in triggers[:10]])
        intents_json = ",".join(f'"{i}"' for i in intents[:5])
        
        content = MANIFEST_TEMPLATE.format(
            plugin_name=plugin_name,
            class_name=class_name,
            description=command[:100],
            triggers=triggers_json,
            intents=intents_json
        )
        
        filepath = os.path.join(plugin_dir, "manifest.json")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def list_generated_plugins(self) -> list:
        """List all generated plugins."""
        if not os.path.exists(GENERATED_PLUGINS_DIR):
            return []
        
        plugins = []
        for entry in os.scandir(GENERATED_PLUGINS_DIR):
            if entry.is_dir():
                manifest_path = os.path.join(entry.path, "manifest.json")
                if os.path.exists(manifest_path):
                    try:
                        with open(manifest_path, "r") as f:
                            manifest = json.load(f)
                            plugins.append({
                                "name": manifest.get("name"),
                                "version": manifest.get("version"),
                                "path": entry.path
                            })
                    except Exception:
                        pass
        return plugins

    def delete_plugin(self, plugin_name: str) -> bool:
        """Delete a generated plugin."""
        plugin_dir = os.path.join(GENERATED_PLUGINS_DIR, plugin_name)
        if os.path.exists(plugin_dir):
            try:
                import shutil
                shutil.rmtree(plugin_dir)
                logger.info(f"Deleted plugin: {plugin_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete plugin {plugin_name}: {e}")
        return False


_plugin_generator: Optional[PluginGenerator] = None


def get_plugin_generator(orchestrator=None) -> PluginGenerator:
    global _plugin_generator
    if _plugin_generator is None:
        _plugin_generator = PluginGenerator(orchestrator)
    return _plugin_generator