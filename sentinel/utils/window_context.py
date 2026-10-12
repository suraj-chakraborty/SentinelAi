"""
sentinel/utils/window_context.py
────────────────────────────────
Extracts real-time foreground window metadata and process identity on Windows.
Enables instant awareness of what application and document the user is working on.
"""

import logging
import sys
from typing import Any

logger = logging.getLogger("WindowContext")


def get_active_window_info() -> dict[str, Any]:
    """
    Returns metadata about the currently focused desktop window:
    {
        "title": str,
        "app_name": str,
        "pid": int,
        "rect": (left, top, right, bottom),
        "is_fullscreen": bool
    }
    """
    info: dict[str, Any] = {"title": "Desktop", "app_name": "Explorer", "pid": 0, "rect": None, "is_fullscreen": False}

    if sys.platform != "win32":
        return info

    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return info

        # Get window title
        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        title = buff.value or "Untitled Window"
        info["title"] = title

        # Get Process ID and Process Name
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        info["pid"] = pid.value

        # Get rect
        rect = wintypes.RECT()
        if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            info["rect"] = (rect.left, rect.top, rect.right, rect.bottom)

        # Get Process Executable name via psutil if available
        try:
            import psutil

            proc = psutil.Process(pid.value)
            info["app_name"] = proc.name()
        except Exception:
            info["app_name"] = "UnknownApp"

    except Exception as e:
        logger.debug(f"Failed to query active window: {e}")

    return info
