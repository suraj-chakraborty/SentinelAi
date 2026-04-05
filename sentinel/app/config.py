"""
sentinel/app/config.py
───────────────────────
Centralised application configuration, paths, and constants.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS

APPDATA_DIR = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi")
ENV_PATH = os.path.join(APPDATA_DIR, ".env")

# Populate environment
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH, override=True)
else:
    load_dotenv()

ACCESS_KEY = os.getenv("PVPORCUPINE_PRIVATE_KEY")
SETTINGS_PATH = os.path.join(APPDATA_DIR, "settings.json")
APPS_REGISTRY_PATH = os.path.join(APPDATA_DIR, "apps_registry.json")
MRU_PATH = os.path.join(APPDATA_DIR, "mru.json")
SECRETS_DB = os.path.join(APPDATA_DIR, "secrets.db")
MACROS_PATH = os.path.join(APPDATA_DIR, "macros.json")
NOTES_PATH = os.path.join(APPDATA_DIR, "notes.json")

# Audio storage
REFERENCE_WAV = os.path.join(APPDATA_DIR, "reference.wav")
OWNER_EMBED_PATH = os.path.join(APPDATA_DIR, "owner_embed.npy")
COMMAND_WAV = os.path.join(APPDATA_DIR, "command.wav")
LIVENESS_WAV = os.path.join(APPDATA_DIR, "liveness.wav")

# Assets
PORCUPINE_KEYWORD_PATH = os.path.join(BASE_DIR, "assets", "sounds", "Hey-robert_en_windows_v3_0_0.ppn")

# ── UI Constants (Modern "Jarvis" Light Theme) ──────────────────────────────
MODERN_BG = "#FFFFFF"         # Pure White
MODERN_SURFACE = "#F8FAFC"    # Slate 50
MODERN_ACCENT = "#0EA5E9"     # Sky 500 (Jarvis Blue)
MODERN_GLOW = "#E0F2FE"       # Sky 100
MODERN_TEXT = "#0F172A"       # Slate 900
MODERN_TEXT_MUTED = "#64748B" # Slate 500
MODERN_SUCCESS = "#10B981"    # Emerald 500
MODERN_WARNING = "#F59E0B"    # Amber 500
MODERN_DANGER = "#EF4444"     # Rose 500

MODERN_FONT = ("Segoe UI Variable Display", 10)
MODERN_FONT_BOLD = ("Segoe UI Variable Display", 10, "bold")
MODERN_FONT_LARGE = ("Segoe UI Variable Display", 16, "bold")

# ── App Aliases ─────────────────────────────────────────────────────────────
ALIASES = {
    "vs code": "vscode",
    "code": "vscode",
    "visual studio code": "vscode",
    "google": "google chrome",
    "chrome": "google chrome",
    "ms word": "word",
    "microsoft word": "word",
    "ms excel": "excel",
    "microsoft excel": "excel",
    "power point": "powerpoint",
    "power-point": "powerpoint",
    "photoshop": "photoshop",
    "illustrator": "illustrator",
    "premiere": "premiere",
    "after effects": "aftereffects",
    "after-effects": "aftereffects",
    "onenote": "onenote",
    "outlook": "outlook",
    "teams": "teams",
    "whatsapp": "whatsapp",
    "telegram": "telegram",
    "discord": "discord"
}

# ── Global Settings Logic ───────────────────────────────────────────────────

def ensure_appdata_dir():
    """Create AppData directory if it doesn't exist."""
    try:
        Path(APPDATA_DIR).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def load_settings() -> dict:
    """Load settings from settings.json."""
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return {}
    return {}

def save_settings(data: dict):
    """Save settings to settings.json."""
    ensure_appdata_dir()
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def get_porcupine_model_path():
    """Locate the Porcupine model file."""
    candidates = []
    if getattr(sys, 'frozen', False):
        candidates.append(os.path.join(BASE_DIR, "pvporcupine", "lib", "common", "porcupine_params.pv"))
        candidates.append(os.path.join(BASE_DIR, "pvporcupine", "resources", "porcupine_params.pv"))
    try:
        import pvporcupine as _pv
        pkg_base = os.path.dirname(_pv.__file__)
        candidates.append(os.path.join(pkg_base, "lib", "common", "porcupine_params.pv"))
        candidates.append(os.path.join(pkg_base, "resources", "porcupine_params.pv"))
    except Exception:
        pass
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None
