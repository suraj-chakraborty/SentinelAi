"""
sentinel/commands/ui_automation.py
───────────────────────────────────
Command to expose UI Automation capabilities.
"""

import logging
from typing import Any

from sentinel.automation.uia_automation import get_uia

logger = logging.getLogger("UIAutomationCommand")


class UIAutomationCommand:
    def __init__(self):
        self._uia = get_uia()

    def execute(self, action: str, **kwargs) -> Any:
        """Execute a UI automation action or natural language RPA command."""
        try:
            act_lower = str(action).lower().strip()
            # Handle natural language RPA requests
            if any(k in act_lower for k in ("snap", "tile", "organize", "center window")):
                import os

                from sentinel.automation.deep_rpa import DeepRPA

                rpa = DeepRPA()
                if "snap left" in act_lower or "tile left" in act_lower or "left" in act_lower:
                    rpa.snap_active_window("left")
                    return "Snapped window to the left half."
                elif "snap right" in act_lower or "tile right" in act_lower or "right" in act_lower:
                    rpa.snap_active_window("right")
                    return "Snapped window to the right half."
                elif "center" in act_lower:
                    rpa.snap_active_window("center")
                    return "Centered active window."
                elif "organize" in act_lower:
                    d_path = os.path.join(os.path.expanduser("~"), "Downloads")
                    res = rpa.organize_directory(d_path)
                    return f"Organized {res.get('files_organized', 0)} files in your Downloads folder."

            if action == "window_info":
                return self._get_window_info()

            elif action == "list_children":
                parent_hwnd = kwargs.get("parent_hwnd")
                children = self._uia.list_child_windows(parent_hwnd)
                return {"success": True, "children": children, "count": len(children)}

            elif action == "find_element":
                name = kwargs.get("name", "")
                hwnd = self._uia.find_element_by_name(name)
                if hwnd:
                    return {"success": True, "handle": hwnd, "name": name}
                return {"success": False, "error": f"Element not found: {name}"}

            elif action == "click":
                hwnd = kwargs.get("handle")
                double_click = kwargs.get("double_click", False)
                if not hwnd:
                    return {"success": False, "error": "handle required"}
                if self._uia.click_element(hwnd, double_click):
                    return {"success": True, "message": f"Clicked element {hwnd}"}
                return {"success": False, "error": "Click failed"}

            elif action == "send_keys":
                text = kwargs.get("text", "")
                hwnd = kwargs.get("handle")
                if not text:
                    return {"success": False, "error": "text required"}
                if self._uia.send_keys(text, hwnd):
                    return {"success": True, "message": f"Sent keys: {text[:50]}..."}
                return {"success": False, "error": "Send keys failed"}

            elif action == "list_windows":
                windows = self._uia.get_all_windows()
                return {"success": True, "windows": windows, "count": len(windows)}

            elif action == "close_window":
                hwnd = kwargs.get("handle")
                if not hwnd:
                    return {"success": False, "error": "handle required"}
                if self._uia.close_window(hwnd):
                    return {"success": True, "message": f"Closed window {hwnd}"}
                return {"success": False, "error": "Close failed"}

            elif action == "minimize":
                hwnd = kwargs.get("handle")
                if not hwnd:
                    return {"success": False, "error": "handle required"}
                if self._uia.minimize_window(hwnd):
                    return {"success": True, "message": f"Minimized window {hwnd}"}
                return {"success": False, "error": "Minimize failed"}

            elif action == "maximize":
                hwnd = kwargs.get("handle")
                if not hwnd:
                    return {"success": False, "error": "handle required"}
                if self._uia.maximize_window(hwnd):
                    return {"success": True, "message": f"Maximized window {hwnd}"}
                return {"success": False, "error": "Maximize failed"}

            elif action == "focus":
                hwnd = kwargs.get("handle")
                if not hwnd:
                    return {"success": False, "error": "handle required"}
                if self._uia.set_focus(hwnd):
                    return {"success": True, "message": f"Focused window {hwnd}"}
                return {"success": False, "error": "Focus failed"}

            else:
                return {"success": False, "error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error(f"UIA action failed: {e}")
            return {"success": False, "error": str(e)}

    def _get_window_info(self) -> dict[str, Any]:
        info = self._uia.get_window_info()
        if "error" in info:
            return {"success": False, "error": info["error"]}
        return {"success": True, "window": info}

    def get_foreground_window(self) -> dict[str, Any]:
        hwnd = self._uia.get_foreground_window()
        if hwnd:
            info = self._uia.get_window_info(hwnd)
            return {"success": True, "handle": hwnd, "window": info}
        return {"success": False, "error": "No foreground window"}


_uia_command: UIAutomationCommand | None = None


def get_uia_command() -> UIAutomationCommand:
    global _uia_command
    if _uia_command is None:
        _uia_command = UIAutomationCommand()
    return _uia_command
