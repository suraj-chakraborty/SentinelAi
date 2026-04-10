"""
sentinel/commands/translate.py
───────────────────────────────
Translate Command - Translates text without opening browser.

Handles "translate hello in spanish", "translate X to Y" commands.
"""

import logging
import threading
from typing import Optional, Dict, Any

logger = logging.getLogger("TranslateCommand")


class Translate:
    """
    Translate text using AI without opening browser.
    """

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._languages = {
            "spanish": "Spanish",
            "french": "French",
            "german": "German",
            "japanese": "Japanese",
            "chinese": "Chinese",
            "korean": "Korean",
            "italian": "Italian",
            "portuguese": "Portuguese",
            "russian": "Russian",
            "hindi": "Hindi",
            "arabic": "Arabic"
        }

    def execute(self, command: str, entity: Optional[str] = None) -> str:
        """
        Execute translation without opening browser.
        """
        # Parse the translation request
        text, target_lang = self._parse_translation(command)
        
        if not text:
            return "What would you like me to translate?"
        
        if not target_lang:
            return "Which language should I translate to?"
        
        # Translate using AI (in background to keep responsive)
        return self._translate_async(text, target_lang)

    def _parse_translation(self, command: str) -> tuple:
        """Parse translation command to extract text and target language."""
        command = command.lower()
        
        # Find target language
        target_lang = None
        for lang_key, lang_name in self._languages.items():
            if lang_key in command:
                target_lang = lang_name
                break
        
        # Extract text to translate
        text = ""
        
        # Patterns like "translate hello in spanish"
        if "translate " in command:
            parts = command.split("translate ")[1]
            if " in " in parts:
                text = parts.split(" in ")[0].strip()
            elif " to " in parts:
                text = parts.split(" to ")[0].strip()
            else:
                text = parts.strip()
        
        # Clean up
        text = text.strip('"').strip("'")
        
        return text, target_lang

    def _translate_async(self, text: str, target_lang: str) -> str:
        """Translate using AI in background."""
        # Speak early acknowledgment
        self._speak(f"Translating to {target_lang}...")
        
        try:
            if self.orchestrator and hasattr(self.orchestrator, "_safe_llm_call"):
                prompt = f"Translate the following text to {target_lang}. Only provide the translation, nothing else.\n\nText: {text}"
                result = self.orchestrator._safe_llm_call(prompt)
                
                if result:
                    # Speak the result
                    self._speak(f"The {target_lang} translation is: {result}")
                    return f"Translation: {result}"
            
            return "Sorry, I couldn't perform the translation."
        
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return f"Sorry, translation failed: {e}"

    def _speak(self, text: str):
        """Speak without blocking."""
        try:
            from sentinel.app.voice import speak
            speak(text, block=False)
        except:
            pass


_translate_command: Optional[Translate] = None


def get_translate_command() -> Translate:
    global _translate_command
    if _translate_command is None:
        _translate_command = Translate()
    return _translate_command