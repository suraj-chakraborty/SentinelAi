"""
sentinel/core/repair.py
────────────────────────
Autonomous Error Self-Repair Agent.
Analyzes failures and suggests corrections using the LLM.
"""

import logging
from typing import Optional, Dict

logger = logging.getLogger("SentinelRepair")

class RepairAgent:
    """
    Analyzes execution failures (Python errors, tool timeouts) 
    and generates correction suggestions.
    """
    
    def __init__(self, llm_callback):
        self.llm = llm_callback

    def analyze_failure(self, command: str, error: str, context: Optional[str] = None) -> Dict[str, str]:
        """
        Takes a failed command and its error, returns a suggestion.
        """
        logger.info(f"Analyzing failure: {error[:50]}...")
        
        prompt = f"""
        System Error Detected in SentinelAI.
        
        Failed Command: {command}
        Error Message: {error}
        Context: {context or "N/A"}
        
        You are the Autonomous Self-Repair Agent. 
        Analyze why this failed (e.g. syntax error, missing dependency, incorrect path, or logic error).
        
        Provide:
        1. A brief explanation of the cause.
        2. A corrected version of the command or code.
        
        Format your response as:
        CAUSE: <explanation>
        CORRECTION: <corrected_command>
        """
        
        response = self.llm(prompt)
        
        # Parse response
        cause = "Unknown"
        correction = ""
        
        for line in response.splitlines():
            if line.startswith("CAUSE:"):
                cause = line.replace("CAUSE:", "").strip()
            elif line.startswith("CORRECTION:"):
                correction = line.replace("CORRECTION:", "").strip()
                
        return {
            "cause": cause,
            "correction": correction or response # Fallback to full response if parsing fails
        }

    def suggest_repair(self, command: str, error: str) -> Optional[str]:
        """High-level entry point to get a repair string for the user."""
        analysis = self.analyze_failure(command, error)
        if analysis["correction"]:
            return f"I encountered an error: {analysis['cause']}. I suggest trying: {analysis['correction']}"
        return None
