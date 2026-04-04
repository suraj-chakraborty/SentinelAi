"""
Close Windows applications without importing sentinel_ai.close_app (no speak/status_queue side effects).
Uses psutil when available, then taskkill, then optional pygetwindow title match.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger("AppCloser")


def _normalize(name: str) -> str:
    return (name or "").lower().strip()


def _resolve_exe_path(name: str) -> Optional[str]:
    """Best-effort path to the app's .exe using the main app's registry index if present."""
    try:
        import sentinel_ai as main

        fn = getattr(main, "find_app_executable", None)
        if callable(fn):
            path = fn(name)
            if path and os.path.isfile(str(path).strip(' "\'')):
                return str(path).strip(' "\'')
    except Exception as e:
        logger.debug("find_app_executable unavailable: %s", e)
    return None


def _candidate_image_names(name: str, exe_path: Optional[str]) -> list[str]:
    """Executable filenames to pass to taskkill /IM."""
    out: list[str] = []
    seen: Set[str] = set()

    def add(im: str) -> None:
        im = im.strip()
        if not im:
            return
        low = im.lower()
        if low not in seen:
            seen.add(low)
            out.append(im)

    if exe_path:
        add(os.path.basename(exe_path))

    base = name.strip()
    if base.lower().endswith(".exe"):
        add(base)
    else:
        add(f"{base}.exe")

    n = _normalize(base).replace(" ", "")
    if n and n + ".exe" not in seen:
        add(f"{n}.exe")

    # Common voice targets
    common = {
        "chrome": "chrome.exe",
        "google chrome": "chrome.exe",
        "edge": "msedge.exe",
        "microsoft edge": "msedge.exe",
        "firefox": "firefox.exe",
        "notepad": "notepad.exe",
        "calculator": "Calculator.exe",
        "calc": "Calculator.exe",
        "spotify": "Spotify.exe",
        "discord": "Discord.exe",
        "code": "Code.exe",
        "vscode": "Code.exe",
        "visual studio code": "Code.exe",
        "terminal": "WindowsTerminal.exe",
        "powershell": "powershell.exe",
        "cmd": "cmd.exe",
    }
    key = _normalize(base)
    if key in common:
        add(common[key])

    return out


def _kill_via_psutil(name: str, exe_path: Optional[str]) -> int:
    try:
        import psutil
    except ImportError:
        return 0

    target = _normalize(name)
    stems: Set[str] = {target}
    if exe_path:
        stems.add(_normalize(Path(exe_path).stem))

    killed = 0
    for proc in psutil.process_iter(["name", "exe"]):
        try:
            info = proc.info or {}
            pname = _normalize(info.get("name") or "")
            pexe = info.get("exe") or ""
            pstem = _normalize(Path(pexe).stem) if pexe else pname
            if not pname and not pstem:
                continue
            if pname in stems or pstem in stems:
                proc.terminate()
                killed += 1
                continue
            if any(t and t in pname for t in stems if len(t) >= 3):
                proc.terminate()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception as e:
            logger.debug("psutil skip: %s", e)
    return killed


def _taskkill_image(im: str) -> bool:
    flags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        flags = subprocess.CREATE_NO_WINDOW
    try:
        r = subprocess.run(
            ["taskkill", "/IM", im, "/F", "/T"],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=flags,
        )
        return r.returncode == 0
    except Exception as e:
        logger.debug("taskkill %s: %s", im, e)
        return False


def _close_windows_by_title(substring: str) -> int:
    if len(substring) < 2:
        return 0
    try:
        import pygetwindow as gw
    except ImportError:
        return 0

    needle = substring.lower()
    closed = 0
    for w in gw.getAllWindows():
        title = (w.title or "").strip().lower()
        if not title or needle not in title:
            continue
        try:
            w.close()
            closed += 1
        except Exception:
            continue
    return closed


def close_application(name: str) -> bool:
    """
    Terminate processes matching the given app name or window title.
    Returns True if at least one close/kill attempt likely succeeded.
    """
    name = (name or "").strip()
    if not name:
        return False

    exe_path = _resolve_exe_path(name)
    killed = _kill_via_psutil(name, exe_path)
    if killed > 0:
        logger.info("Closed via psutil: %s (%s process(es))", name, killed)
        return True

    for im in _candidate_image_names(name, exe_path):
        if _taskkill_image(im):
            logger.info("Closed via taskkill: %s (%s)", name, im)
            return True

    # Last resort: window title (e.g. user said "close the essay document")
    if _close_windows_by_title(_normalize(name)):
        logger.info("Closed via window title match: %s", name)
        return True

    logger.warning("Could not close: %s", name)
    return False
