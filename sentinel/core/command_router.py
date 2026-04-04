import logging
import os
import importlib.util
from typing import Dict, Any, Optional

from sentinel.core.preprocessor import preprocess_command
from sentinel.core.intent_detector import IntentDetector, Intent
from sentinel.memory.context_manager import ContextManager

logger = logging.getLogger("CommandRouter")

# Intent enum values often differ from plugin module names (e.g. shutdown → system_control).
INTENT_TO_PLUGIN: Dict[Intent, str] = {
    Intent.OPEN_APP: "open_app",
    Intent.CLOSE_APP: "close_app",
    Intent.SHUTDOWN: "system_control",
    Intent.RESTART: "system_control",
    Intent.SYSTEM_STATUS: "system_control",
    Intent.DEACTIVATE: "deactivate",
    Intent.AUTONOMOUS_AGENT: "ai_agent",
}

class CommandRouter:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.preprocessor = preprocess_command
        self.detector = IntentDetector()
        self.context = ContextManager(os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi", "context.json"))
        self.plugins: Dict[str, Any] = {}
        self._load_core_commands()
        self._plugin_system = None
        self._load_manifest_plugins()

    def _load_manifest_plugins(self):
        """Load optional manifest-based plugins from AppData and built-in dirs."""
        try:
            from sentinel.core.plugin_system import PluginSystem

            self._plugin_system = PluginSystem(orchestrator=self.orchestrator)
            n = self._plugin_system.load_all()
            if n:
                logger.info("Manifest plugin system: %s extension(s) loaded.", n)
        except Exception as e:
            logger.debug("Manifest plugins not loaded: %s", e)
            self._plugin_system = None

    def _load_core_commands(self):
        """Loads core command plugins from sentinel/commands/."""
        commands_dir = os.path.join(os.path.dirname(__file__), "..", "commands")
        if not os.path.exists(commands_dir):
            logger.error(f"Commands directory not found: {commands_dir}")
            return

        for filename in os.listdir(commands_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                plugin_name = filename[:-3]
                module_path = os.path.join(commands_dir, filename)
                try:
                    spec = importlib.util.spec_from_file_location(plugin_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Assuming each plugin has a class named after its file in PascalCase
                    # e.g., open_app.py -> OpenApp
                    class_name = "".join(word.title() for word in plugin_name.split("_"))
                    if hasattr(module, class_name):
                        plugin_class = getattr(module, class_name)
                        self.plugins[plugin_name] = plugin_class(self.orchestrator)
                        logger.info(f"Loaded core command plugin: {plugin_name}")
                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_name}: {e}")

    def run_command(self, raw_command: str) -> Optional[str]:
        """
        Processes a raw command and routes it to the appropriate plugin or AI.
        """
        # 1. Preprocess
        cleaned = self.preprocessor(raw_command)
        if not cleaned:
            return None

        # 2. Detect Intent
        intent = self.detector.detect_intent(cleaned)
        entity = self.detector.extract_entity(cleaned, intent)
        
        # 3. Update Context
        self.context.update_context("command", raw_command)
        self.context.update_context("intent", intent.value)
        if entity:
            self.context.update_context("entity", entity)
        self.context.save()

        logger.info(f"Routing command: '{cleaned}' | Intent: {intent.name} | Entity: {entity}")

        # 4. Route to core plugin (map intent → module name)
        plugin_key = INTENT_TO_PLUGIN.get(intent, intent.value)
        if plugin_key in self.plugins:
            try:
                # Use the plugin name string as key (as it's derived from file name)
                return self.plugins[plugin_key].execute(cleaned, entity)
            except Exception as e:
                logger.error(f"Error executing plugin {plugin_key}: {e}")
                return f"Sorry, I encountered an error while processing that command: {e}"

        # 4b. Manifest-based extensions when no core handler matched this intent
        if self._plugin_system:
            try:
                ext_result = self._plugin_system.dispatch(cleaned)
                if ext_result:
                    return ext_result
            except Exception as e:
                logger.error("Extension plugin dispatch error: %s", e)

        # 5. Fallback: AI (specifically check if it's an autonomous goal)
        if intent == Intent.AUTONOMOUS_AGENT or intent == Intent.UNKNOWN:
            # If unknown but looks like a complex task, try autonomous agent
            return self.ai_fallback(cleaned)

        return None

    def ai_fallback(self, command: str) -> str:
        """
        Handles commands not recognized by the router by sending them to the AI agent.
        """
        # Decisions on whether this should be an autonomous goal are handled here
        # or in the ai_agent.py plugin.
        if "ai_agent" in self.plugins:
            return self.plugins["ai_agent"].execute(command, command)
            
        # Last resort fallback to raw LLM call
        if self.orchestrator and hasattr(self.orchestrator, "_safe_llm_call"):
            return self.orchestrator._safe_llm_call(command)
            
        return "I'm not sure how to handle that command yet."
