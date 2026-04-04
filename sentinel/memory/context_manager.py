import logging
import json
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("ContextManager")

class ContextManager:
    def __init__(self, storage_path: str = None):
        self.context: Dict[str, Any] = {
            "last_app": None,
            "last_intent": None,
            "history": [],
            "user_preferences": {}
        }
        self.storage_path = storage_path
        if self.storage_path and os.path.exists(self.storage_path):
            self.load()

    def update_context(self, key: str, value: Any):
        self.context[key] = value
        if key == "intent":
             self.context["last_intent"] = value
        elif key == "entity" and self.context.get("last_intent") == "open_app":
             self.context["last_app"] = value
        
        # Keep a rolling history of the last 10 commands
        if key == "command":
            self.context["history"].append(value)
            if len(self.context["history"]) > 10:
                self.context["history"].pop(0)

    def get_context(self, key: str) -> Any:
        return self.context.get(key)

    def save(self):
        if self.storage_path:
            try:
                with open(self.storage_path, "w") as f:
                    json.dump(self.context, f)
            except Exception as e:
                logger.error(f"Failed to save context: {e}")

    def load(self):
        if self.storage_path and os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    self.context.update(json.load(f))
            except Exception as e:
                logger.error(f"Failed to load context: {e}")
