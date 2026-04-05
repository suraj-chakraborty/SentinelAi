"""
Notes Plugin — sentinel/plugins/notes/plugin.py
────────────────────────────────────────────────
Create, read and list personal notes stored as a JSON file.

Trigger examples:
  "take a note: buy groceries"
  "note that the meeting is at 3pm"
  "jot down call dentist tomorrow"
  "read my notes"
  "show all notes"
  "clear notes"
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import List, Optional

from sentinel.core.plugin_system import PluginBase

logger = logging.getLogger("NotesPlugin")

APPDATA_DIR  = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi")
_NOTES_PATH  = os.path.join(APPDATA_DIR, "notes.json")

_TAKE_RE = re.compile(
    r'\b(?:take\s+a?\s*note|note\s+that|jot\s+(?:this\s+)?down|'
    r'write\s+(?:this\s+)?down|add\s+a?\s*note|remember)\s*[:\-]?\s*(.+)',
    re.IGNORECASE,
)


class NotesPlugin(PluginBase):
    """Persistent personal notes stored as JSON in AppData."""

    def can_handle(self, command: str) -> bool:
        cmd = command.lower()
        return any(kw in cmd for kw in (
            "note", "jot", "write down", "my notes", "show notes",
            "read notes", "list notes", "clear notes", "delete notes"
        ))

    def handle(self, command: str) -> str:
        cmd = command.lower()
        if any(kw in cmd for kw in ("read", "show", "list", "what are my")):
            return self._read_notes()
        if any(kw in cmd for kw in ("clear", "delete all", "remove all")):
            return self._clear_notes()
        return self._take_note(command)

    def on_load(self):
        os.makedirs(APPDATA_DIR, exist_ok=True)
        logger.info("NotesPlugin loaded. Notes file: %s", _NOTES_PATH)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _take_note(self, command: str) -> str:
        match = _TAKE_RE.search(command)
        if match:
            content = match.group(1).strip()
        else:
            # Strip known filler and use the whole command
            content = re.sub(
                r'\b(take a note|note that|jot down|remember|write down|add a note)[:\-]?\s*',
                '', command, flags=re.IGNORECASE,
            ).strip()
            if not content:
                return "What would you like me to note down?"

        notes = self._load()
        note = {"id": len(notes) + 1, "content": content, "timestamp": time.strftime("%Y-%m-%d %H:%M")}
        notes.append(note)
        self._save(notes)
        return f"Noted: \"{content}\""

    def _read_notes(self) -> str:
        notes = self._load()
        if not notes:
            return "You have no notes yet. Say 'take a note' to create one."
        
        # Filter out any non-dictionary items that might have corrupted the list
        valid_notes = [n for n in notes if isinstance(n, dict)]
        if not valid_notes:
            return "I couldn't find any valid notes. Say 'take a note' to create one."
            
        lines = [f"You have {len(valid_notes)} note(s):"]
        for n in valid_notes[-10:]:    # show last 10 to avoid huge TTS output
            lines.append(f"  [{n.get('timestamp', 'N/A')}] {n.get('content', 'Empty content')}")
        return "\n".join(lines)

    def _clear_notes(self) -> str:
        count = len(self._load())
        self._save([])
        return f"Cleared {count} note(s)."

    def _load(self) -> List[dict]:
        try:
            if os.path.exists(_NOTES_PATH):
                with open(_NOTES_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as exc:
            logger.error("Failed to load notes: %s", exc)
        return []

    def _save(self, notes: List[dict]) -> None:
        try:
            with open(_NOTES_PATH, "w", encoding="utf-8") as f:
                json.dump(notes, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error("Failed to save notes: %s", exc)
