"""
sentinel/app/registry.py
─────────────────────────
Windows application registry, fuzzy matching, and MRU tracking.
"""

import os
import winreg
import json
import logging
import subprocess
import pythoncom
import win32com.client
from pathlib import Path
from typing import Dict, Optional, List

from sentinel.app.config import (
    APPS_REGISTRY_PATH, MRU_PATH, ALIASES, APPDATA_DIR,
    ensure_appdata_dir
)

logger = logging.getLogger("SentinelRegistry")

# ── Normalize & Index ────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """Lowercase and strip a name for indexing."""
    return (name or "").lower().strip()

def load_apps_index() -> dict:
    """Load the apps index from JSON."""
    try:
        if os.path.exists(APPS_REGISTRY_PATH):
            with open(APPS_REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return {}
    return {}

def save_apps_index(index: dict):
    """Save the apps index to JSON."""
    ensure_appdata_dir()
    try:
        with open(APPS_REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
    except Exception:
        pass

# ── MRU (Most Recently Used) ────────────────────────────────────────────────

def load_mru() -> dict:
    """Load MRU data."""
    try:
        if os.path.exists(MRU_PATH):
            with open(MRU_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return {}
    return {}

def save_mru(mru: dict):
    """Save MRU data."""
    ensure_appdata_dir()
    try:
        with open(MRU_PATH, "w", encoding="utf-8") as f:
            json.dump(mru, f, indent=2)
    except Exception:
        pass

def record_mru(name: str):
    """Increment use count for an app."""
    n = normalize_name(name)
    mru = load_mru()
    rec = mru.get(n, {"count": 0})
    rec["count"] = int(rec.get("count", 0)) + 1
    mru[n] = rec
    save_mru(mru)

def get_top_mru(limit=10) -> List[str]:
    """Get list of most used app names."""
    mru = load_mru()
    items = sorted(mru.items(), key=lambda kv: kv[1].get("count", 0), reverse=True)
    return [k for k, _ in items[:limit]]

# ── Registry Scanning ───────────────────────────────────────────────────────

def _scan_uninstall_key(root) -> Dict[str, str]:
    """Scan a Windows Uninstall registry key for display names and icons/exes."""
    results = {}
    try:
        with winreg.OpenKey(root, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall") as key:
            i = 0
            while True:
                try:
                    sub = winreg.EnumKey(key, i)
                except OSError:
                    break
                i += 1
                try:
                    with winreg.OpenKey(key, sub) as sk:
                        name, _ = winreg.QueryValueEx(sk, "DisplayName")
                        if not name: continue
                        
                        icon = None
                        loc = None
                        try: icon, _ = winreg.QueryValueEx(sk, "DisplayIcon")
                        except OSError: pass
                        try: loc, _ = winreg.QueryValueEx(sk, "InstallLocation")
                        except OSError: pass
                        
                        exe = None
                        if icon:
                            c = str(icon).split(',')[0].strip(' "\'')
                            if c.lower().endswith(".exe") and os.path.exists(c):
                                exe = c
                        
                        if not exe and loc and os.path.isdir(loc):
                            norm_name = normalize_name(name)
                            candidates = list(Path(loc).glob("*.exe"))
                            if candidates:
                                for cand in candidates:
                                    if norm_name in cand.stem.lower():
                                        exe = str(cand)
                                        break
                                if not exe:
                                    exe = str(candidates[0])
                        
                        if name:
                            results[normalize_name(name)] = exe or ""
                except (OSError, ValueError):
                    continue
    except OSError:
        pass
    return results

def scan_start_menu_apps() -> Dict[str, str]:
    """Scan start menu shortcuts for executables."""
    try:
        pythoncom.CoInitialize()
        shell = win32com.client.Dispatch("WScript.Shell")
    except Exception:
        return {}
    
    paths = [
        os.path.expandvars(r"%ProgramData%\Microsoft\Windows\Start Menu\Programs"),
        os.path.expandvars(r"%AppData%\Microsoft\Windows\Start Menu\Programs")
    ]
    apps = {}
    for base in paths:
        if not os.path.isdir(base): continue
        for root, dirs, files in os.walk(base):
            for f in files:
                if f.lower().endswith(".lnk"):
                    lnk_path = os.path.join(root, f)
                    try:
                        shortcut = shell.CreateShortCut(lnk_path)
                        target = shortcut.Targetpath
                        if target and target.lower().endswith(".exe") and os.path.exists(target):
                            name = f[:-4]
                            apps[normalize_name(name)] = target
                    except Exception:
                        pass
    return apps

def build_app_registry(fast_only=False) -> dict:
    """Rebuild the entire application registry."""
    idx = {}
    idx.update(_scan_uninstall_key(winreg.HKEY_LOCAL_MACHINE))
    idx.update(_scan_uninstall_key(winreg.HKEY_CURRENT_USER))
    
    if not fast_only:
        try:
            # Note: scan_windows_app_aliases appears to be missing or a typo in original.
            # We'll stick to Start Menu for now.
            idx.update(scan_start_menu_apps())
        except Exception:
            pass

    # Fallback for known common apps
    known = {
        "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "google chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "notepad": r"C:\Windows\System32\notepad.exe",
        "paint": r"C:\Windows\System32\mspaint.exe",
        "calculator": r"C:\Windows\System32\calc.exe",
        "vscode": os.path.expandvars(r"%LocalAppData%\Programs\Microsoft VS Code\Code.exe"),
        "edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "spotify": os.path.expandvars(r"%LocalAppData%\Microsoft\WindowsApps\Spotify.exe"),
    }
    for k, v in known.items():
        if k not in idx and os.path.exists(v):
            idx[k] = v
            
    save_apps_index(idx)
    return idx

# ── Search & Match ──────────────────────────────────────────────────────────

def _fuzzy_match(query: str, target: str) -> float:
    """Simple fuzzy score."""
    q = normalize_name(query)
    t = normalize_name(target)
    if q == t: return 1.0
    if q in t: return 0.8 + (len(q) / len(t)) * 0.15
    qw = set(q.split())
    tw = set(t.split())
    if qw.intersection(tw):
        return 0.5 + (len(qw.intersection(tw)) / len(qw)) * 0.3
    return 0.0

def find_app_executable(name: str) -> Optional[str]:
    """Find the full path to an app's executable."""
    n = normalize_name(name)
    if n in ALIASES:
        n = ALIASES[n]
        
    index = load_apps_index()
    if not index:
        index = build_app_registry(fast_only=True)
    
    # 1. Exact
    exe = index.get(n)
    if exe and os.path.exists(str(exe).strip(' "\'')):
        return str(exe).strip(' "\'')
        
    # 2. Fuzzy
    best_match = None
    best_score = 0.65
    for k, v in index.items():
        if not v or not os.path.exists(str(v).strip(' "\'')): continue
        score = _fuzzy_match(n, k)
        if score > best_score:
            best_score = score
            best_match = str(v).strip(' "\'')
            
    return best_match
