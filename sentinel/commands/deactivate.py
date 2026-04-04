import logging
import time

logger = logging.getLogger("Deactivate")

class Deactivate:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def execute(self, command: str, entity: str) -> str:
        """
        Deactivates and closes the Sentinel application.
        """
        logger.info("Deactivate plugin triggered.")
        
        # We trigger the shutdown in a small delay to allow the response to be spoken/returned
        import threading
        def _delayed_shutdown():
            time.sleep(2)
            self.orchestrator.shutdown()
            
        threading.Thread(target=_delayed_shutdown, daemon=True).start()
        
        return "Deactivating Sentinel systems. Goodbye, Master."
