import logging
import os

logger = logging.getLogger("CloseApp")

class CloseApp:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def execute(self, command: str, app_name: str) -> str:
        """
        Logic for terminating an application process.
        """
        if not app_name:
            return "What application would you like me to close?"

        try:
            from sentinel.utils.app_closer import close_application

            if close_application(app_name):
                return f"Closed {app_name}."
            return f"Sorry, I couldn't find {app_name}."
        except Exception as e:
            logger.error(f"Error in CloseApp plugin: {e}")
            return f"Failed to close {app_name}: {e}"
