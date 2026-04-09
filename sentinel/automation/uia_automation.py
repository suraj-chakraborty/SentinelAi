"""
sentinel/automation/uia_automation.py
─────────────────────────────────────
Windows UI Automation (UIA) wrapper for accessibility automation.
Provides element discovery, inspection, and action execution.
"""

import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger("UIAutomation")

try:
    import win32gui
    import win32con
    import win32api
    import pywintypes
    _WIN32_OK = True
except ImportError:
    _WIN32_OK = False
    logger.warning("win32gui/pywin32 not available - UIA limited")


@dataclass
class UIElement:
    """Represents a UI element with its properties."""
    handle: int
    name: str
    role: str
    class_name: str
    rect: Tuple[int, int, int, int]
    is_enabled: bool
    is_visible: bool


class UIAutomation:
    """Windows UI Automation wrapper."""

    def __init__(self):
        self._current_window = None

    def get_foreground_window(self) -> Optional[int]:
        """Get the handle of the currently active window."""
        if not _WIN32_OK:
            return None
        try:
            return win32gui.GetForegroundWindow()
        except Exception as e:
            logger.error(f"Failed to get foreground window: {e}")
            return None

    def get_window_title(self, hwnd: Optional[int] = None) -> str:
        """Get the title of a window."""
        if not _WIN32_OK:
            return ""
        if hwnd is None:
            hwnd = self.get_foreground_window()
        if hwnd is None:
            return ""
        try:
            return win32gui.GetWindowText(hwnd)
        except Exception:
            return ""

    def get_window_info(self, hwnd: Optional[int] = None) -> Dict[str, Any]:
        """Get detailed information about a window."""
        if not _WIN32_OK:
            return {"error": "win32 not available"}
        
        if hwnd is None:
            hwnd = self.get_foreground_window()
        if hwnd is None:
            return {"error": "No foreground window"}
        
        try:
            rect = win32gui.GetWindowRect(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            title = win32gui.GetWindowText(hwnd)
            pid = win32api.GetWindowThreadProcessId(hwnd)[0]
            
            return {
                "handle": hwnd,
                "title": title,
                "class_name": class_name,
                "rect": {"left": rect[0], "top": rect[1], "right": rect[2], "bottom": rect[3]},
                "process_id": pid,
                "visible": win32gui.IsWindowVisible(hwnd) == 1
            }
        except Exception as e:
            return {"error": str(e)}

    def list_child_windows(self, parent_hwnd: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all child windows of a parent window."""
        if not _WIN32_OK:
            return []
        
        if parent_hwnd is None:
            parent_hwnd = self.get_foreground_window()
        if parent_hwnd is None:
            return []
        
        children = []
        
        def callback(hwnd, extra):
            try:
                if win32gui.IsWindowVisible(hwnd):
                    name = win32gui.GetWindowText(hwnd)
                    class_name = win32gui.GetClassName(hwnd)
                    if name or class_name:
                        children.append({
                            "handle": hwnd,
                            "name": name,
                            "class_name": class_name
                        })
            except Exception:
                pass
        
        try:
            win32gui.EnumChildWindows(parent_hwnd, callback, None)
        except Exception as e:
            logger.error(f"Failed to enumerate child windows: {e}")
        
        return children

    def find_element_by_name(self, name: str, parent_hwnd: Optional[int] = None) -> Optional[int]:
        """Find a window/element by its name (partial match)."""
        if not _WIN32_OK:
            return None
        
        if parent_hwnd is None:
            parent_hwnd = self.get_foreground_window()
        if parent_hwnd is None:
            return None
        
        def callback(hwnd, extra):
            try:
                if win32gui.IsWindowVisible(hwnd):
                    text = win32gui.GetWindowText(hwnd)
                    if name.lower() in text.lower():
                        extra.append(hwnd)
            except Exception:
                pass
        
        matches = []
        try:
            win32gui.EnumChildWindows(parent_hwnd, callback, matches)
        except Exception:
            pass
        
        return matches[0] if matches else None

    def click_element(self, hwnd: int, double_click: bool = False) -> bool:
        """Click on a UI element."""
        if not _WIN32_OK:
            return False
        
        try:
            rect = win32gui.GetWindowRect(hwnd)
            x = (rect[0] + rect[2]) // 2
            y = (rect[1] + rect[3]) // 2
            
            win32api.SetCursorPos((x, y))
            time.sleep(0.05)
            
            if double_click:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
                time.sleep(0.05)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
            else:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
            
            return True
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return False

    def set_focus(self, hwnd: int) -> bool:
        """Bring a window to foreground and focus."""
        if not _WIN32_OK:
            return False
        
        try:
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.1)
            return True
        except Exception as e:
            logger.error(f"Set focus failed: {e}")
            return False

    def get_element_at_position(self, x: int, y: int) -> Optional[int]:
        """Get the window handle at a specific screen position."""
        if not _WIN32_OK:
            return None
        
        try:
            return win32gui.WindowFromPoint(win32api.GetPixel(win32api.GetDC(0), x, y))
        except Exception:
            pass
        
        try:
            return win32gui.WindowFromPoint((x, y))
        except Exception:
            return None

    def send_keys(self, text: str, hwnd: Optional[int] = None) -> bool:
        """Send keystrokes to a window."""
        if not _WIN32_OK:
            return False
        
        try:
            if hwnd is None:
                hwnd = self.get_foreground_window()
            if hwnd is None:
                return False
            
            self.set_focus(hwnd)
            time.sleep(0.1)
            
            for char in text:
                win32api.SendMessage(hwnd, win32con.WM_CHAR, ord(char), 0)
                time.sleep(0.02)
            
            return True
        except Exception as e:
            logger.error(f"Send keys failed: {e}")
            return False

    def get_all_windows(self) -> List[Dict[str, Any]]:
        """Get all visible windows on the system."""
        if not _WIN32_OK:
            return []
        
        windows = []
        
        def callback(hwnd, extra):
            try:
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    class_name = win32gui.GetClassName(hwnd)
                    if title:
                        windows.append({
                            "handle": hwnd,
                            "title": title,
                            "class_name": class_name
                        })
            except Exception:
                pass
        
        try:
            win32gui.EnumWindows(callback, None)
        except Exception as e:
            logger.error(f"Failed to enumerate windows: {e}")
        
        return windows

    def close_window(self, hwnd: int) -> bool:
        """Close a window."""
        if not _WIN32_OK:
            return False
        
        try:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            return True
        except Exception as e:
            logger.error(f"Close window failed: {e}")
            return False

    def minimize_window(self, hwnd: int) -> bool:
        """Minimize a window."""
        if not _WIN32_OK:
            return False
        
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            return True
        except Exception as e:
            logger.error(f"Minimize window failed: {e}")
            return False

    def maximize_window(self, hwnd: int) -> bool:
        """Maximize a window."""
        if not _WIN32_OK:
            return False
        
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            return True
        except Exception as e:
            logger.error(f"Maximize window failed: {e}")
            return False

    def restore_window(self, hwnd: int) -> bool:
        """Restore a minimized window."""
        if not _WIN32_OK:
            return False
        
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            return True
        except Exception as e:
            logger.error(f"Restore window failed: {e}")
            return False


_uia_instance: Optional[UIAutomation] = None


def get_uia() -> UIAutomation:
    global _uia_instance
    if _uia_instance is None:
        _uia_instance = UIAutomation()
    return _uia_instance