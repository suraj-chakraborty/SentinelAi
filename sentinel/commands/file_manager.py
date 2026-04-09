import logging
import os
import re

logger = logging.getLogger("FileManager")


class FileManager:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    def _path_from_command(self, text: str) -> str:
        if not text:
            return ""
        # crude extraction: take after keywords
        for kw in ("create file", "write file", "open file", "read file", "find file"):
            if kw in text:
                return text.split(kw, 1)[1].strip(" '\"")
        return text.strip()

    def execute(self, command: str, entity: str) -> str:
        text = (command or "").lower()
        if not text:
            return "What file operation would you like me to perform?"

        if ("create file" in text) or ("write file" in text):
            path = self._path_from_command(text)
            if not path:
                return "Please specify a file path."
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write("")
                return f"Created file at {path}"
            except Exception as e:
                return f"Failed to create file: {e}"

        if "read file" in text:
            path = self._path_from_command(text)
            if not path or not os.path.exists(path):
                return f"File not found: {path}"
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()[:1000]
            except Exception as e:
                return f"Failed to read file: {e}"

        if "find file" in text:
            pattern = text.split("find file", 1)[1].strip()
            if not pattern:
                pattern = "**/*"
            import glob
            matches = glob.glob(pattern, recursive=True)
            return "\n".join(matches[:20]) if matches else "No files found."

        return f"File operation not recognized: {entity or command}"
