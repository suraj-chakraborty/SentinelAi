"""
sentinel/commands/git_copilot.py
────────────────────────────────
Voice Command Plugin for Git and IDE Copilot.
Handles "commit changes", "git status", and "analyze terminal error".
"""

import logging
from typing import Optional, Dict, Any

from sentinel.automation.ide_watcher import IDEWatcher

logger = logging.getLogger("GitCopilotCommand")


class GitCopilotCommand:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.watcher = IDEWatcher()

    def execute(self, command: str, entity: Optional[str] = None) -> str:
        cmd_lower = command.lower()

        # 1. Commit changes
        if any(k in cmd_lower for k in ("commit", "commit changes", "save changes to git")):
            hint = entity or None
            # If command contains "message ...", extract it
            if "message" in cmd_lower:
                hint = command.split("message", 1)[1].strip()
            
            res = self.watcher.git_smart_commit(message_hint=hint)
            if res.get("success"):
                return f"Committed successfully with message: '{res['commit_message']}'. {res.get('stat', '')}"
            return f"Git commit failed: {res.get('message') or res.get('error')}"

        # 2. Check Git status
        if any(k in cmd_lower for k in ("git status", "git changes", "what files changed")):
            status = self.watcher.git_status_summary()
            if not status.get("is_git"):
                return "The current workspace is not a Git repository."
            count = status.get("change_count", 0)
            if count == 0:
                return "Working tree is clean. No uncommitted changes."
            return f"You have {count} modified file(s). Top changes: {', '.join(status['changes'][:3])}."

        # 3. Analyze terminal error / stack trace
        if any(k in cmd_lower for k in ("terminal error", "why did my build fail", "stack trace", "build error")):
            # Get clipboard or active context
            try:
                import tkinter as tk
                root = tk.Tk()
                root.withdraw()
                clip_text = root.clipboard_get()
                root.destroy()
            except Exception:
                clip_text = ""

            if clip_text and len(clip_text) > 10:
                analysis = self.watcher.analyze_stack_trace(clip_text)
                return f"Found error from clipboard: {analysis['summary']}."
            
            context = self.watcher.get_dev_context()
            return f"Active editor is {context['app_name']} focusing on '{context['window_title']}'."

        return "Git copilot command not understood."
