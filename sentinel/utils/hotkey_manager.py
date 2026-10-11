"""
sentinel/utils/hotkey_manager.py
────────────────────────────────
System-wide Global Hotkey Manager for Windows.
Allows triggering Sentinel instantly anywhere using Alt+J or Ctrl+Shift+Space.
"""

import sys
import threading
import logging
from typing import Callable, Optional

logger = logging.getLogger("GlobalHotkey")


class GlobalHotkeyManager:
    """
    Hooks global keyboard shortcuts on Windows using ctypes Win32 message pump.
    """

    # Virtual key codes & modifiers
    MOD_ALT = 0x0001
    MOD_CONTROL = 0x0002
    MOD_SHIFT = 0x0004
    MOD_WIN = 0x0008
    VK_J = 0x4A
    VK_SPACE = 0x20

    def __init__(self, on_trigger: Optional[Callable[[], None]] = None):
        self.on_trigger = on_trigger
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start hotkey listener in background thread."""
        if sys.platform != "win32" or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._msg_loop, daemon=True, name="HotkeyPump")
        self._thread.start()
        logger.info("GlobalHotkeyManager registered (Alt+J / Ctrl+Shift+J).")

    def stop(self):
        """Stop hotkey pump."""
        self._running = False
        if sys.platform == "win32" and self._thread:
            try:
                import ctypes
                # Post WM_QUIT to thread
                ctypes.windll.user32.PostQuitMessage(0)
            except Exception:
                pass
            self._thread = None

    def _msg_loop(self):
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            HOTKEY_ID_ALT_J = 101
            HOTKEY_ID_CTRL_SHIFT_J = 102
            HOTKEY_ID_WIN_SPACE = 103
            HOTKEY_ID_CTRL_SHIFT_SPACE = 104

            # Register Alt + J
            if not user32.RegisterHotKey(None, HOTKEY_ID_ALT_J, self.MOD_ALT, self.VK_J):
                logger.debug("Failed to register Alt+J hotkey (may already be in use)")

            # Register Ctrl + Shift + J as fallback
            if not user32.RegisterHotKey(None, HOTKEY_ID_CTRL_SHIFT_J, self.MOD_CONTROL | self.MOD_SHIFT, self.VK_J):
                logger.debug("Failed to register Ctrl+Shift+J hotkey")

            # Register Win + Space
            if not user32.RegisterHotKey(None, HOTKEY_ID_WIN_SPACE, self.MOD_WIN, self.VK_SPACE):
                logger.debug("Win+Space in use by Windows; trying Ctrl+Shift+Space")

            # Register Ctrl + Shift + Space
            if not user32.RegisterHotKey(None, HOTKEY_ID_CTRL_SHIFT_SPACE, self.MOD_CONTROL | self.MOD_SHIFT, self.VK_SPACE):
                logger.debug("Failed to register Ctrl+Shift+Space hotkey")

            msg = wintypes.MSG()
            while self._running:
                # Peek or GetMessage
                ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if ret == 0 or ret == -1:
                    break

                if msg.message == 0x0312:  # WM_HOTKEY
                    hk_id = msg.wParam
                    if hk_id in (HOTKEY_ID_ALT_J, HOTKEY_ID_CTRL_SHIFT_J, HOTKEY_ID_WIN_SPACE, HOTKEY_ID_CTRL_SHIFT_SPACE):
                        logger.info(f"Global hotkey triggered (id={hk_id})!")
                        if self.on_trigger:
                            try:
                                self.on_trigger()
                            except Exception as cb_err:
                                logger.error(f"Error in hotkey callback: cb_err={cb_err}")

                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))

            # Unregister on exit
            user32.UnregisterHotKey(None, HOTKEY_ID_ALT_J)
            user32.UnregisterHotKey(None, HOTKEY_ID_CTRL_SHIFT_J)
            user32.UnregisterHotKey(None, HOTKEY_ID_WIN_SPACE)
            user32.UnregisterHotKey(None, HOTKEY_ID_CTRL_SHIFT_SPACE)

        except Exception as e:
            logger.debug(f"Hotkey loop exception: {e}")
