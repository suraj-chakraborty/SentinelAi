"""
sentinel/app/utils.py
─────────────────────
General utility functions for file management and system maintenance.
"""

import os
import time
import json
import logging
from sentinel.app.config import NOTES_PATH, ensure_appdata_dir
from sentinel.app.state import get_state

logger = logging.getLogger("SentinelUtils")

# ── Notes Management ────────────────────────────────────────────────────────

def add_note(text: str) -> bool:
    """Save a user note to persistent JSON storage."""
    ensure_appdata_dir()
    notes = []
    try:
        if os.path.exists(NOTES_PATH):
            with open(NOTES_PATH, "r", encoding="utf-8") as f:
                notes = json.load(f)
    except Exception:
        notes = []
        
    notes.append({
        "timestamp": time.time(),
        "content": text
    })
    
    try:
        with open(NOTES_PATH, "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=2)
        return True
    except Exception:
        return False

def list_notes() -> list:
    """Retrieve all saved notes."""
    try:
        if os.path.exists(NOTES_PATH):
            with open(NOTES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return []
    return []

# ── System Maintenance ──────────────────────────────────────────────────────

def clean_downloads_folder(days=30) -> int:
    """Delete files in the Downloads folder older than [days] days."""
    try:
        path = os.path.join(os.path.expanduser("~"), "Downloads")
        count = 0
        threshold = time.time() - (86400 * days)
        
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                if os.path.getmtime(fp) < threshold:
                    os.remove(fp)
                    count += 1
        return count
    except Exception as e:
        logger.error(f"Error cleaning downloads: {e}")
        return 0

def find_large_files_in_downloads(threshold_mb=100) -> list:
    """Find files in Downloads larger than [threshold_mb] MB."""
    try:
        path = os.path.join(os.path.expanduser("~"), "Downloads")
        large_files = []
        threshold_bytes = threshold_mb * 1024 * 1024
        
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                size = os.path.getsize(fp)
                if size > threshold_bytes:
                    large_files.append((f, size / (1024 * 1024)))
        
        large_files.sort(key=lambda x: x[1], reverse=True)
        return large_files
    except Exception as e:
        logger.error(f"Error finding large files: {e}")
        return []
