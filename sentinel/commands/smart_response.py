"""
sentinel/commands/smart_response.py
───────────────────────────────
Smart Response Command - Answer without opening apps.

Handles simple Q&A, calculations, definitions without browser/app launch.
"""

import logging
import re
import math
from typing import Optional

logger = logging.getLogger("SmartResponse")


class SmartResponse:
    """
    Answer questions and perform tasks WITHOUT opening apps/browsers.
    """

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    def execute(self, command: str, entity: Optional[str] = None) -> str:
        """
        Execute smart response without launching apps.
        """
        cmd = command.lower().strip()
        
        # Calculator
        if any(k in cmd for k in ["calculate", "what is", "compute", "how much", "plus", "minus", "times", "divided"]):
            return self._handle_calculator(cmd)
        
        # Definition/Explanation (simple Q&A - no app needed)
        if any(cmd.startswith(p) for p in ["what is ", "who is ", "what are ", "define ", "explain "]):
            return self._handle_simple_qa(cmd)
        
        # Translation is handled by translate command
        
        # Weather - just tell, don't open browser
        if any(k in cmd for k in ["weather", "temperature", "forecast"]):
            return self._handle_weather_simple(cmd)
        
        # Default - let AI answer but without executing
        return self._handle_ai_fallback(command)

    def _handle_calculator(self, cmd: str) -> str:
        """Handle math calculations locally."""
        try:
            # Extract numbers and operators
            expr = cmd.replace("calculate", "").replace("what is", "").replace("how much is", "")
            expr = expr.replace("plus", "+").replace("minus", "-").replace("times", "*").replace("multiplied by", "*")
            expr = expr.replace("divided by", "/").replace("divide by", "/").replace("over", "/")
            expr = expr.replace("equals", "=").replace("equal to", "=")
            
            # Clean expression
            expr = re.sub(r'[^\d+\-*/().]', '', expr)
            
            if expr:
                result = eval(expr)
                return f"The answer is {result}"
        except:
            pass
        
        # Fall back to AI
        return self._handle_ai_fallback(cmd)

    def _handle_simple_qa(self, cmd: str) -> str:
        """Handle simple questions without opening anything."""
        if self.orchestrator and hasattr(self.orchestrator, "_safe_llm_call"):
            try:
                prompt = f"Give a brief, direct answer. Don't suggest opening anything.\n\nQuestion: {cmd}"
                result = self.orchestrator._safe_llm_call(prompt)
                if result:
                    return result
            except:
                pass
        
        return "I'm not sure about that."

    def _handle_weather_simple(self, cmd: str) -> str:
        """Simple weather response without opening browser."""
        # Use cached/simple response
        if self.orchestrator and hasattr(self.orchestrator, "_safe_llm_call"):
            try:
                prompt = "Give a brief weather update. Don't suggest opening anything. If you don't have weather data, just say so."
                result = self.orchestrator._safe_llm_call(prompt)
                if result:
                    return result
            except:
                pass
        
        return "I don't have access to current weather data. You could check your weather app."

    def _handle_ai_fallback(self, command: str) -> str:
        """Handle via AI but DON'T execute any app/browser commands."""
        if self.orchestrator and hasattr(self.orchestrator, "_safe_llm_call"):
            try:
                # Explicitly tell AI NOT to open anything
                prompt = f"""Answer the following question directly. 
Do NOT suggest opening any app, website, or browser.
Just provide the answer in text.

Question: {command}"""
                result = self.orchestrator._safe_llm_call(prompt)
                if result:
                    return result
            except Exception as e:
                logger.error(f"AI fallback error: {e}")
        
        return "I'm not sure how to answer that."


_smart_response: Optional[SmartResponse] = None


def get_smart_response() -> SmartResponse:
    global _smart_response
    if _smart_response is None:
        _smart_response = SmartResponse()
    return _smart_response