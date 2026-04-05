import logging
from sentinel.app.voice import speak

logger = logging.getLogger("Deactivate")

class Deactivate:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def execute(self, command: str, entity: str) -> str:
        """
        Deactivates and closes the Sentinel application.
        """
        logger.info("Deactivate plugin triggered.")
        
        # Speak the goodbye message and wait for it to finish
        message = "Deactivating Sentinel systems. Goodbye, Master."
        speak(message, block=True)
        
        # Shutdown the orchestrator and exit
        self.orchestrator.shutdown()
        
        return "System Deactivated."
