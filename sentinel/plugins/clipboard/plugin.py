"""
Clipboard Plugin — sentinel/plugins/clipboard/plugin.py
────────────────────────────────────────────────────────
Read and write the Windows clipboard via win32clipboard.

Trigger examples:
  "what did I copy"
  "read my clipboard"
  "copy this to clipboard: Hello World"
  "paste"
"""

from __future__ import annotations

import logging
import re

from sentinel.core.plugin_system import PluginBase

logger = logging.getLogger("ClipboardPlugin")

_WRITE_RE = re.compile(
    r'\b(?:copy\s+(?:this\s+)?to\s+clipboard|save\s+to\s+clipboard|'
    r'add\s+to\s+clipboard)[:\-]?\s*(.+)',
    re.IGNORECASE,
)


def _get_clipboard() -> str:
    """Read current clipboard text."""
    try:
        import win32clipboard
        win32clipboard.OpenClipboard()
        try:
            return win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
        finally:
            win32clipboard.CloseClipboard()
    except Exception as exc:
        logger.warning("Clipboard read error: %s", exc)
        # Fallback: tkinter
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            text = root.clipboard_get()
            root.destroy()
            return text
        except Exception:
            return ""


def _set_clipboard(text: str) -> bool:
    """Write text to the clipboard."""
    try:
        import win32clipboard
        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, text)
        finally:
            win32clipboard.CloseClipboard()
        return True
    except Exception as exc:
        logger.warning("Clipboard write (win32) error: %s", exc)
        try:
            import subprocess
            subprocess.run("clip", input=text.encode("utf-16-le"), check=True)
            return True
        except Exception as exc2:
            logger.error("Clipboard write (clip) error: %s", exc2)
            return False


class ClipboardPlugin(PluginBase):
    """Read/write the Windows clipboard."""

    def can_handle(self, command: str) -> bool:
        cmd = command.lower()
        return any(kw in cmd for kw in (
            "clipboard", "what did i copy", "paste", "copy this"
        ))

    def handle(self, command: str) -> str:
        match = _WRITE_RE.search(command)
        if match:
            text = match.group(1).strip()
            success = _set_clipboard(text)
            return f"Copied to clipboard: \"{text[:80]}\"" if success else "Failed to write to clipboard."

        # Read
        content = _get_clipboard()
        if not content:
            return "The clipboard is empty or contains non-text content."
        preview = content[:300]
        if len(content) > 300:
            preview += f"… ({len(content)} chars total)"
        return f"Clipboard contains:\n{preview}"

    def on_load(self):
        logger.info("ClipboardPlugin loaded.")
