"""
sentinel/automation/ide_watcher.py
──────────────────────────────────
Terminal & IDE Watcher.
Inspects active code editors, terminal stack traces, and provides Git automation.
"""

import logging
import os
import re
import subprocess
from typing import Any

from sentinel.utils.window_context import get_active_window_info

logger = logging.getLogger("IDEWatcher")


class IDEWatcher:
    """
    Watches the active development environment (VS Code, Windows Terminal, PowerShell)
    and provides contextual code analysis and automated Git workflows.
    """

    def __init__(self, workspace_root: str | None = None):
        self.workspace_root = workspace_root or os.getcwd()

    def get_dev_context(self) -> dict[str, Any]:
        """Returns the current developer context from the active window."""
        win_info = get_active_window_info()
        title = win_info.get("title", "")
        app_name = win_info.get("app_name", "").lower()

        is_ide = any(k in app_name for k in ("code", "pycharm", "sublime", "devenv", "notepad"))
        is_terminal = any(k in app_name for k in ("terminal", "powershell", "cmd", "bash", "conhost"))

        # Extract file name if in title (e.g. "ide_watcher.py - SentinelAi - Visual Studio Code")
        active_file = None
        if " - " in title:
            candidate = title.split(" - ")[0].strip()
            if "." in candidate and " " not in candidate:
                active_file = candidate

        return {
            "app_name": win_info.get("app_name", ""),
            "window_title": title,
            "is_ide": is_ide,
            "is_terminal": is_terminal,
            "active_file": active_file,
        }

    def analyze_stack_trace(self, trace_text: str) -> dict[str, Any]:
        """Parses a raw terminal stack trace to identify the root error and file."""
        lines = trace_text.strip().split("\n")
        error_type = "Unknown Error"
        error_file = None
        error_line = None

        # Python traceback pattern: File "...", line X, in ...
        py_file_match = re.findall(r'File "([^"]+)", line (\d+)', trace_text)
        if py_file_match:
            error_file, error_line = py_file_match[-1]

        for line in reversed(lines):
            line_s = line.strip()
            if any(err in line_s for err in ("Error:", "Exception:", "FAILED", "AssertionError", "SyntaxError")):
                error_type = line_s
                break

        return {
            "error_type": error_type,
            "file": error_file,
            "line": error_line,
            "summary": f"{error_type} in {os.path.basename(error_file) if error_file else 'terminal'}",
        }

    def git_status_summary(self, repo_dir: str | None = None) -> dict[str, Any]:
        """Returns modified, untracked, and staged files for the workspace repository."""
        target_dir = repo_dir or self.workspace_root
        try:
            res = subprocess.run(
                ["git", "status", "--porcelain"], cwd=target_dir, capture_output=True, text=True, check=False
            )
            if res.returncode != 0:
                return {"is_git": False, "changes": []}

            changes = [line.strip() for line in res.stdout.strip().split("\n") if line.strip()]
            return {"is_git": True, "change_count": len(changes), "changes": changes[:10]}
        except Exception as e:
            logger.debug(f"Git status error: {e}")
            return {"is_git": False, "error": str(e)}

    def git_smart_commit(self, message_hint: str | None = None, repo_dir: str | None = None) -> dict[str, Any]:
        """Stages all changes, creates a conventional commit message, and commits."""
        target_dir = repo_dir or self.workspace_root
        try:
            # 1. Stage changes
            subprocess.run(["git", "add", "-A"], cwd=target_dir, check=True)

            # 2. Get staged diff summary
            diff = subprocess.run(
                ["git", "diff", "--cached", "--stat"], cwd=target_dir, capture_output=True, text=True, check=True
            ).stdout.strip()

            if not diff:
                return {"success": False, "message": "No changes staged to commit."}

            # 3. Formulate commit message
            if message_hint:
                msg = message_hint.strip()
                if not any(
                    msg.startswith(prefix) for prefix in ("feat:", "fix:", "refactor:", "docs:", "chore:", "test:")
                ):
                    msg = f"feat: {msg}"
            else:
                msg = "feat: implement automated updates and improvements"

            # 4. Commit
            res = subprocess.run(
                ["git", "commit", "-m", msg], cwd=target_dir, capture_output=True, text=True, check=True
            )

            return {
                "success": True,
                "commit_message": msg,
                "stat": diff.split("\n")[-1] if diff else "",
                "commit_output": res.stdout.strip() if res.stdout else "",
            }
        except Exception as e:
            logger.error(f"Git smart commit error: {e}")
            return {"success": False, "error": str(e)}
