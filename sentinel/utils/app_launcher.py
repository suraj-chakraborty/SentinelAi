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
        if not base: continue
        path = os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")
        if os.path.isfile(path): return path
    return shutil.which("chrome.exe") or shutil.which("chrome")

def _edge_exe() -> Optional[str]:
    for env_key in ("PROGRAMFILES(X86)", "PROGRAMFILES"):
        base = os.environ.get(env_key)
        if not base: continue
        path = os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe")
        if os.path.isfile(path): return path
    return shutil.which("msedge.exe") or shutil.which("msedge")

def _notepad_exe() -> Optional[str]:
    # Notepad is standard Windows
    path = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "notepad.exe")
    if os.path.isfile(path): return path
    return shutil.which("notepad.exe") or shutil.which("notepad")

def _photoshop_exe() -> Optional[str]:
    # Check common Adobe paths
    for env_key in ("PROGRAMFILES", "PROGRAMFILES(X86)"):
        base = os.environ.get(env_key)
        if not base: continue
        adobe_root = os.path.join(base, "Adobe")
        if not os.path.isdir(adobe_root): continue
        # Look for any folder contains Photoshop
        for folder in os.listdir(adobe_root):
            if "photoshop" in folder.lower():
                exe = os.path.join(adobe_root, folder, "Photoshop.exe")
                if os.path.isfile(exe): return exe
    return shutil.which("photoshop.exe") or shutil.which("photoshop")


def launch_application(name: str) -> bool:
    """
    Try to start an application by display name or executable name.
    Returns True if a launch was attempted without fatal error (may still fail silently).
    """
    name = (name or "").strip()
    if not name:
        return False

    target = name.lower()
    
    # ── Hardcoded Path Resolution ─────────────────────────────────────────────
    def get_standard_path(app_id: str) -> Optional[str]:
        prog64 = os.environ.get("PROGRAMFILES", "C:\\Program Files")
        prog86 = os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)")
        win_dir = os.environ.get("SystemRoot", "C:\\Windows")
        
        paths = {
            "chrome": [
                os.path.join(prog64, "Google", "Chrome", "Application", "chrome.exe"),
                os.path.join(prog86, "Google", "Chrome", "Application", "chrome.exe")
            ],
            "edge": [
                os.path.join(prog86, "Microsoft", "Edge", "Application", "msedge.exe"),
                os.path.join(prog64, "Microsoft", "Edge", "Application", "msedge.exe")
            ],
            "brave": [
                os.path.join(prog64, "BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
                os.path.join(prog86, "BraveSoftware", "Brave-Browser", "Application", "brave.exe")
            ],
            "notepad": [
                os.path.join(win_dir, "notepad.exe"),
                os.path.join(win_dir, "System32", "notepad.exe")
            ]
        }
        
        for candidate in paths.get(app_id, []):
            if os.path.isfile(candidate):
                return candidate
        return None

    exe_map = {
        "chrome": "chrome",
        "google chrome": "chrome",
        "edge": "edge",
        "msedge": "edge",
        "microsoft edge": "edge",
        "brave": "brave",
        "notepad": "notepad",
        "photoshop": "photoshop",
        "adobe photoshop": "photoshop",
    }

    if target in exe_map:
        app_id = exe_map[target]
        if app_id == "photoshop":
            exe = _photoshop_exe()
        else:
            exe = get_standard_path(app_id) or globals().get(f"_{app_id}_exe", lambda: None)()
            
        if exe:
            try:
                # Use quoted path to handle spaces
                subprocess.Popen(f'"{exe}"', shell=True)
                return True
            except OSError as e:
                logger.warning(f"{target} launch failed: {e}")

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

    # Use os.startfile only if the path actually exists to avoid system popups
    if os.path.exists(name):
        try:
            os.startfile(name)
            return True
        except OSError:
            pass

    # Final attempt: Shell execution via subprocess, but WITHOUT 'start' to avoid popups
    try:
        subprocess.Popen(name, shell=True)
        return True
    except Exception:
        pass

    return False
