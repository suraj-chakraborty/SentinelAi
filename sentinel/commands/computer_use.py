"""
sentinel/commands/computer_use.py
──────────────────────────────────
Core command plugin for desktop automation.
"""

import logging
from sentinel.app.voice import speak

logger = logging.getLogger("ComputerUsePlugin")

class ComputerUse:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def execute(self, command: str, entity: str = None) -> str:
        """
        Trigger the ComputerUseAgent for desktop automation.
        """
        agent = self.orchestrator.computer_use
        vision = self.orchestrator.vision_module

        # Fallback to pure vision if agent is missing or it's a simple 'describe' request
        if not agent or "describe" in command or "analyze" in command or "what is on" in command:
            if vision:
                logger.info("Using VisionModule fallback for screen analysis.")
                try:
                    analysis = vision.analyze_screen("Describe what is on the screen in detail.")
                    return analysis if analysis else "I can see the screen, but I'm having trouble describing it."
                except Exception as e:
                    logger.error(f"Vision fallback failed: {e}")
            
            if not agent:
                return "Computer Use Agent is not initialized, and Vision fallback also failed."

        if agent.is_active:
            return "The Computer Use Agent is already executing a task. Please wait."

        # The 'entity' usually contains the core instruction for the agent.
        goal = entity or command
        logger.info(f"Triggering ComputerUseAgent with goal: {goal}")
        
        # Run in a background thread to avoid blocking the orchestrator
        import threading
        def _run():
            try:
                result = agent.execute(goal)
                speak(f"Computer use task finished: {result}")
            except Exception as e:
                logger.error(f"ComputerUse Agent crashed: {e}")
                speak("I encountered an error during computer use automation.")

        threading.Thread(target=_run, daemon=True).start()
        
        return f"Initiating desktop automation for: {goal}. Please do not use the mouse or keyboard while I work."
