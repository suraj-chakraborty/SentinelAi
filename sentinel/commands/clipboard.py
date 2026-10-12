import logging

logger = logging.getLogger("Clipboard")

try:
    import pyperclip
except Exception:
    pyperclip = None


class Clipboard:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    def execute(self, command: str, entity: str) -> str:
        if not command:
            return "What would you like me to do with the clipboard?"
        text = (command or "").lower()
        if "read clipboard" in text:
            if pyperclip:
                return pyperclip.paste()
            return "Clipboard access not available (pyperclip not installed)."
        if "copy to clipboard" in text or "write to clipboard" in text:
            content = entity or command
            if pyperclip:
                pyperclip.copy(content)
                return "Clipboard updated."
            return "Clipboard write not available (pyperclip not installed)."
        return "Clipboard command not recognised."
