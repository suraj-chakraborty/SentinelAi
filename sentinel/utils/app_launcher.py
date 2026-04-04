"""
Launch local Windows applications without importing sentinel_ai (avoids circular imports).
Falls back to sentinel_ai.open_app when the main app has built the apps index.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger("AppLauncher")


def _chrome_exe() -> Optional[str]:
    for env_key in ("PROGRAMFILES", "PROGRAMFILES(X86)"):
        base = os.environ.get(env_key)
        if not base:
            continue
        path = os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")
        if os.path.isfile(path):
            return path
    return shutil.which("chrome") or shutil.which("chrome.exe")


def launch_application(name: str) -> bool:
    """
    Try to start an application by display name or executable name.
    Returns True if a launch was attempted without fatal error (may still fail silently).
    """
    name = (name or "").strip()
    if not name:
        return False

    target = name.lower()

    # Prefer real Chrome when user asks for Chrome (not a web search URL).
    if target in ("chrome", "google chrome"):
        exe = _chrome_exe()
        if exe:
            try:
                subprocess.Popen([exe], shell=False)
                return True
            except OSError as e:
                logger.warning("Chrome launch failed: %s", e)

    # Optional: full registry index from main entrypoint
    try:
        import sentinel_ai as _main

        if hasattr(_main, "open_app"):
            return bool(_main.open_app(name))
    except Exception as e:
        logger.debug("sentinel_ai.open_app unavailable: %s", e)

    exe = shutil.which(name) or shutil.which(f"{name}.exe")
    if exe:
        try:
            subprocess.Popen([exe], shell=False)
            return True
        except OSError as e:
            logger.warning("which() launch failed: %s", e)

    try:
        os.startfile(name)
        return True
    except OSError:
        pass

    try:
        flags = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            flags = subprocess.CREATE_NO_WINDOW
        subprocess.run(
            ["cmd", "/c", "start", "", name],
            shell=False,
            check=False,
            creationflags=flags,
        )
        return True
    except Exception as e:
        logger.warning("start-command launch failed: %s", e)
        return False
