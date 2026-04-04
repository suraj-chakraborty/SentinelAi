#
# sentinel_ai_fixed.py
import os
import sys
import time
import wave
import threading
import webbrowser
import subprocess
import sqlite3
import hashlib
import hmac
import struct
from queue import Queue
from dotenv import load_dotenv

# ── SUPPRESS PYSTRAY CTYPES SPAM ──────────────────────────────────────────
# pystray WNDPROC events bypass standard stderr and use the unraisablehook handler in py3.8+
import logging
class _SuppressWNDPROC(logging.Filter):
    def filter(self, record):
        return "WNDPROC" not in record.getMessage()

def custom_unraisablehook(unraisable):
    # Silence specific noisy but harmless errors from Windows libraries
    err_str = str(unraisable.exc_value)
    if unraisable.exc_type is TypeError:
        # Pystray and some older Windows libs throw unraisable TypeErrors with no message 
        # or about LRESULT/WPARAM casting during event handling.
        if not err_str or any(x in err_str for x in ["WPARAM", "LRESULT", "WNDPROC", "HWND"]):
            return
            
    # For actually useful unraisable errors, use the default handler
    try:
        sys.__unraisablehook__(unraisable)
    except Exception:
        pass

sys.unraisablehook = custom_unraisablehook

# Also try to redirect stderr for these specific messages if they are coming from ctypes directly
_original_stderr = sys.stderr
class StderrFilter:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        # Even more aggressive filtering for raw terminal output
        if not data or not data.strip():
            return
            
        # Silence persistent noise from Windows event handlers (PyStray, Tkinter, etc.)
        # Sometimes these come in chunks like 'TypeError:', ': ', or just ':'
        noise_keywords = ["TypeError", "WNDPROC", "LRESULT", "WPARAM", "HWND"]
        if any(x in data for x in noise_keywords) or data.strip() == ":":
            return # Silent drop for these specific known noise-makers
                
        self.stream.write(data)
    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass

sys.stderr = StderrFilter(_original_stderr)
# ────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────

# Initialize Tier 4 structured logging
import logging
try:
    import sentinel.core.logger
except ImportError:
    pass

# audio / VAD / embeddings
import pvporcupine
import pyaudio
import numpy as np
import soundfile as sf

# TTS & STT
import pyttsx3
import speech_recognition as sr
import pythoncom
import httpx
import shutil
import winsound
try:
    import requests
except Exception:
    requests = None
import json
import winreg
try:
    import psutil
except Exception:
    psutil = None
from pathlib import Path
import logging

# Speaker embedding utilities
# Silero VAD utils
 

# GUI
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
try:
    import pystray
    from PIL import Image, ImageDraw
    _tray_available = True
except Exception:
    _tray_available = False

load_dotenv()
# Also load from AppData if exists (for packaged EXE)
APPDATA_DIR = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi")
env_path = os.path.join(APPDATA_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

def ensure_env_setup(force=False):
    """Checks for required API keys and shows a wizard if they are missing."""
    required = {
        "PVPORCUPINE_PRIVATE_KEY": "Picovoice Access Key",
        "GEMINI_API_KEY": "Gemini API Key",
        "OPENROUTER_API_KEY": "OpenRouter API Key (Optional)"
    }
    
    # Only block if critical keys are missing
    critical = ["PVPORCUPINE_PRIVATE_KEY", "GEMINI_API_KEY"]
    missing_critical = [k for k in critical if not os.getenv(k)]
    
    if not missing_critical and not force:
        return True
    if missing_critical and not force:
        return False

    # Show Wizard
    root = tk.Tk()
    root.title("SentinelAI | First Run Setup")
    root.geometry("450x350")
    root.configure(bg="#1e293b")
    
    # Style
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TLabel", background="#1e293b", foreground="#f1f5f9", font=("Inter", 10))
    style.configure("Header.TLabel", font=("Inter", 14, "bold"), foreground="#22d3ee")
    
    ttk.Label(root, text="Welcome to SentinelAI", style="Header.TLabel").pack(pady=20)
    ttk.Label(root, text="Please provide your API keys to continue.\nThese will be stored securely in your AppData folder.", justify="center").pack(pady=10)

    entries = {}
    for key, label in required.items():
        frame = ttk.Frame(root, style="TFrame")
        frame.pack(fill="x", padx=40, pady=10)
        ttk.Label(frame, text=label).pack(anchor="w")
        entry = ttk.Entry(frame, width=40, show="*" if "KEY" in key else "")
        entry.insert(0, os.getenv(key, ""))
        entry.pack(pady=5)
        entries[key] = entry

    def save_and_close():
        os.makedirs(APPDATA_DIR, exist_ok=True)
        with open(env_path, "w") as f:
            for key, entry in entries.items():
                val = entry.get().strip()
                if val:
                    f.write(f"{key}={val}\n")
                    os.environ[key] = val
        
        # Refresh global configuration with new keys
        refresh_config()
                
        root.destroy()

    ttk.Button(root, text="Save & Start Sentinel", command=save_and_close).pack(pady=30)
    
    root.protocol("WM_DELETE_WINDOW", lambda: os._exit(0)) # Exit if closed without saving
    root.mainloop()
    return True

# ---------- Configuration ----------
wake_thread = None
wake_stop_event = threading.Event()

# refresh_config and start_wake_listener moved below listen_for_wake_word_loop to avoid NameError

# App registry paths
APPDATA_DIR = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi")

ACCESS_KEY = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS

PORCUPINE_KEYWORD_PATH = os.path.join(BASE_DIR, "assets", "sounds", "Hey-robert_en_windows_v3_0_0.ppn")
def _porcupine_model_path():
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
REFERENCE_WAV = os.path.join(APPDATA_DIR, "reference.wav")
OWNER_EMBED_PATH = os.path.join(APPDATA_DIR, "owner_embed.npy")
COMMAND_WAV = os.path.join(APPDATA_DIR, "command.wav")
LIVENESS_WAV = os.path.join(APPDATA_DIR, "liveness.wav")

sentinel_orchestrator = None
porcupine_status = "Waiting..."
# refresh_config() # Initial load - MOVED TO main() to avoid NameError during startup

# status queue for GUI updates
status_queue = Queue()

# Global status variables for Brain health
brain_status = "CONNECTING..."
brain_status_color = "#94a3b8" # Muted
local_status = "INIT..."
local_status_color = "#94a3b8"

def _brain_monitor_loop():
    """Background thread to monitor LLM connectivity and local service health."""
    global brain_status, brain_status_color, local_status, local_status_color
    
    first_run = True
    while True:
        try:
            # 1. Check Local (Ollama) every minute
            if sentinel_orchestrator and hasattr(sentinel_orchestrator, 'ollama_module'):
                ollama = sentinel_orchestrator.ollama_module
                if ollama.is_available():
                    local_status = "ONLINE"
                    local_status_color = "#22c55e" # Green
                elif getattr(ollama, 'is_installing', False):
                    local_status = "INSTALLING..."
                    local_status_color = "#f59e0b" # Orange
                else:
                    local_status = "OFFLINE"
                    local_status_color = "#ef4444" # Red
                    # Auto-start if offline
                    try:
                        ollama.start_service()
                    except Exception:
                        pass
            else:
                local_status = "UNAVAILABLE"
                local_status_color = "#64748b"

            # 2. Check Brain (Gemini/OpenRouter) every 5-10 mins (spare quota)
            # On first run, we check immediately, then relax to 10 minutes.
            current_time = int(time.time())
            if first_run or (current_time % 600 < 40):
                try:
                    # Minimal probe logic - using a very simple prompt
                    # We use the unified gemini_generate which handles the user's manual 3.0-flash
                    resp = gemini_generate("Ping status check. Reply 'OK'.", model="gemini-2.0-flash")
                    
                    if "connecting" in resp.lower() or "internet" in resp.lower():
                        brain_status = "OFFLINE"
                        brain_status_color = "#ef4444" 
                    elif "quota" in resp.lower() or "429" in resp.lower():
                        brain_status = "QUOTA"
                        brain_status_color = "#f59e0b" # Orange
                    elif "not found" in resp.lower() or "404" in resp.lower() or "not a valid model" in resp.lower():
                        brain_status = "ID ERROR" 
                        brain_status_color = "#f43f5e" # Rose
                    else:
                        brain_status = "ONLINE"
                        brain_status_color = "#22c55e" # Green
                except Exception:
                    brain_status = "OFFLINE"
                    brain_status_color = "#ef4444"
                first_run = False
            
        except Exception:
            pass
        time.sleep(40) # Poll periodically

threading.Thread(target=_brain_monitor_loop, daemon=True).start()
tray_icon = None
tray_lock = threading.Lock()

agent_active = False
pending_step = None
next_agent_capture_time = 0
entertainment_active = False

# App registry paths
APPS_REGISTRY_PATH = os.path.join(APPDATA_DIR, "apps_registry.json")
apps_index = {}
_mru = {}
MRU_PATH = os.path.join(APPDATA_DIR, "mru.json")
SETTINGS_PATH = os.path.join(APPDATA_DIR, "settings.json")
wake_stop_event = threading.Event()
wake_thread = None
root_window = None
listening_blocked_until = 0
mic_level_var = None
recording_overlay = None
recording_overlay_var = None
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
SECRETS_DB = os.path.join(APPDATA_DIR, "secrets.db")
MASTER_META_KEY = "master_hash"
MASTER_META_SALT = "master_salt"
MACROS_PATH = os.path.join(APPDATA_DIR, "macros.json")

def _normalize_name(name):
    return (name or "").lower().strip()

def ensure_appdata_dir():
    try:
        Path(APPDATA_DIR).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        log_path = os.path.join(APPDATA_DIR, "sentinel.log")
        # simple rotation
        try:
            if os.path.exists(log_path) and os.path.getsize(log_path) > 1_000_000:
                bak = log_path + ".1"
                try:
                    if os.path.exists(bak):
                        os.remove(bak)
                except Exception:
                    pass
                os.replace(log_path, bak)
        except Exception:
            pass
        logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    except Exception:
        pass

def _load_apps_index():
    global apps_index
    try:
        if os.path.exists(APPS_REGISTRY_PATH):
            with open(APPS_REGISTRY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                apps_index.clear()
                apps_index.update(data)
    except Exception:
        apps_index.clear()

def _load_mru():
    global _mru
    try:
        if os.path.exists(MRU_PATH):
            with open(MRU_PATH, "r", encoding="utf-8") as f:
                _mru = json.load(f)
    except Exception:
        _mru = {}

def _save_mru():
    try:
        with open(MRU_PATH, "w", encoding="utf-8") as f:
            json.dump(_mru, f, indent=2)
    except Exception:
        pass

def record_mru(name):
    n = _normalize_name(name)
    if not _mru:
        _load_mru()
    rec = _mru.get(n, {"count": 0})
    rec["count"] = int(rec.get("count", 0)) + 1
    _mru[n] = rec
    _save_mru()

def top_mru(limit=5):
    if not _mru:
        _load_mru()
    items = sorted(_mru.items(), key=lambda kv: kv[1].get("count", 0), reverse=True)
    return [k for k, _ in items[:limit]]

def _save_apps_index():
    try:
        with open(APPS_REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(apps_index, f, indent=2)
    except Exception:
        pass

def _load_settings():
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return {}
    return {}

def _save_settings(data):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def _ensure_secrets_db():
    ensure_appdata_dir()
    conn = sqlite3.connect(SECRETS_DB)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB)")
    cur.execute("CREATE TABLE IF NOT EXISTS secrets (name TEXT PRIMARY KEY, salt BLOB, nonce BLOB, ciphertext BLOB)")
    conn.commit()
    return conn

def _macros_load():
    ensure_appdata_dir()
    try:
        if os.path.exists(MACROS_PATH):
            with open(MACROS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return {}
    return {}

def _macros_save(macros):
    try:
        with open(MACROS_PATH, "w", encoding="utf-8") as f:
            json.dump(macros, f, indent=2)
        return True
    except Exception:
        return False

def _pbkdf(master, salt, length=32, rounds=200000):
    return hashlib.pbkdf2_hmac('sha256', master.encode('utf-8'), salt, rounds, dklen=length)

def _keystream(key, nonce, length):
    out = bytearray()
    counter = 0
    while len(out) < length:
        counter_bytes = struct.pack('<Q', counter)
        block = hashlib.sha256(key + nonce + counter_bytes).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])

def _encrypt(master, plaintext_bytes):
    salt = os.urandom(16)
    key = _pbkdf(master, salt)
    nonce = os.urandom(16)
    ks = _keystream(key, nonce, len(plaintext_bytes))
    ct = bytes(a ^ b for a, b in zip(plaintext_bytes, ks))
    return salt, nonce, ct

def _decrypt(master, salt, nonce, ciphertext):
    key = _pbkdf(master, salt)
    ks = _keystream(key, nonce, len(ciphertext))
    pt = bytes(a ^ b for a, b in zip(ciphertext, ks))
    return pt

# Optional AES-GCM upgrade if cryptography is available
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:
    AESGCM = None

def _encrypt_aes(master, plaintext_bytes):
    if AESGCM is None:
        return _encrypt(master, plaintext_bytes)
    salt = os.urandom(16)
    key = _pbkdf(master, salt, length=32)
    aes = AESGCM(key)
    nonce = os.urandom(12)
    ct = aes.encrypt(nonce, plaintext_bytes, None)
    return salt, nonce, ct

def _decrypt_aes(master, salt, nonce, ciphertext):
    if AESGCM is None:
        return _decrypt(master, salt, nonce, ciphertext)
    key = _pbkdf(master, salt, length=32)
    aes = AESGCM(key)
    return aes.decrypt(nonce, ciphertext, None)

def _get_master_record(conn):
    cur = conn.cursor()
    cur.execute("SELECT value FROM meta WHERE key=?", (MASTER_META_SALT,))
    row_salt = cur.fetchone()
    cur.execute("SELECT value FROM meta WHERE key=?", (MASTER_META_KEY,))
    row_hash = cur.fetchone()
    return (row_salt[0] if row_salt else None, row_hash[0] if row_hash else None)

def _set_master_record(conn, master):
    salt = os.urandom(16)
    mh = _pbkdf(master, salt)
    cur = conn.cursor()
    cur.execute("REPLACE INTO meta(key,value) VALUES(?,?)", (MASTER_META_SALT, salt))
    cur.execute("REPLACE INTO meta(key,value) VALUES(?,?)", (MASTER_META_KEY, mh))
    conn.commit()

def _verify_master(conn, master):
    if not master:
        return False
    salt, mh = _get_master_record(conn)
    if not salt or not mh:
        _set_master_record(conn, master)
        # Re-fetch the salt that was just set
        salt, mh = _get_master_record(conn)
        if salt and mh and sentinel_orchestrator:
            key = _pbkdf(master, salt)
            sentinel_orchestrator.set_master_key(key)
        return True
    
    key = _pbkdf(master, salt)
    if hmac.compare_digest(key, mh):
        # Update orchestrator master key if available
        if sentinel_orchestrator:
            sentinel_orchestrator.set_master_key(key)
        return True
    return False

def _prompt_master(parent=None):
    try:
        pw = simpledialog.askstring("Master Password", "Enter master password:", show='*', parent=parent)
    except Exception:
        pw = None
    return pw or ""

def store_secret(name, password, parent=None):
    conn = _ensure_secrets_db()
    master = _prompt_master(parent)
    if not master:
        messagebox.showerror("Secrets", "Master password required.")
        conn.close()
        return False
    if not _verify_master(conn, master):
        messagebox.showerror("Secrets", "Invalid master password.")
        conn.close()
        return False
    s = _load_settings()
    if bool(s.get("require_strong_vault", False)) and AESGCM is None:
        messagebox.showerror("Secrets", "Strong vault required but AES-GCM unavailable.")
        conn.close()
        return False
    salt, nonce, ct = _encrypt_aes(master, password.encode('utf-8'))
    cur = conn.cursor()
    cur.execute("REPLACE INTO secrets(name,salt,nonce,ciphertext) VALUES(?,?,?,?)", (name, salt, nonce, ct))
    conn.commit()
    conn.close()
    return True

def fetch_secret(name, parent=None):
    conn = _ensure_secrets_db()
    master = _prompt_master(parent)
    if not master:
        messagebox.showerror("Secrets", "Master password required.")
        conn.close()
        return None
    if not _verify_master(conn, master):
        messagebox.showerror("Secrets", "Invalid master password.")
        conn.close()
        return None
    cur = conn.cursor()
    cur.execute("SELECT salt, nonce, ciphertext FROM secrets WHERE name=?", (name,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    salt, nonce, ct = row
    pt = _decrypt_aes(master, salt, nonce, ct)
    try:
        return pt.decode('utf-8')
    except Exception:
        return None

def speak_password_spelled(pw):
    parts = []
    for ch in pw:
        if ch.isalpha():
            if ch.isupper():
                parts.append(f"capital {ch}")
            else:
                parts.append(f"small {ch}")
        elif ch.isdigit():
            parts.append(f"digit {ch}")
        else:
            names = {
                ' ': 'space', '-': 'dash', '_': 'underscore', '@': 'at', '#': 'hash',
                '!': 'exclamation', '$': 'dollar', '%': 'percent', '^': 'caret', '&': 'ampersand',
                '*': 'asterisk', '(': 'left parenthesis', ')': 'right parenthesis',
                '+': 'plus', '=': 'equals', '[': 'left bracket', ']': 'right bracket',
                '{': 'left brace', '}': 'right brace', ';': 'semicolon', ':': 'colon',
                '"': 'double quote', '\\': 'backslash', '/': 'slash', '?': 'question mark',
                ',': 'comma', '.': 'dot', '<': 'less than', '>': 'greater than', '`': 'backtick',
                '|': 'pipe'
            }
            parts.append(f"symbol {names.get(ch, ch)}")
    speak(" ".join(parts))

def _scan_uninstall_key(root):
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
                        try:
                            icon, _ = winreg.QueryValueEx(sk, "DisplayIcon")
                        except OSError:
                            pass
                        try:
                            loc, _ = winreg.QueryValueEx(sk, "InstallLocation")
                        except OSError:
                            pass
                        
                        exe = None
                        if icon:
                            c = str(icon).split(',')[0].strip(' "\'')
                            if c.lower().endswith(".exe") and os.path.exists(c):
                                exe = c
                        
                        if not exe and loc and os.path.isdir(loc):
                            # Try common executable names matching the display name
                            norm_name = _normalize_name(name)
                            candidates = list(Path(loc).glob("*.exe"))
                            if candidates:
                                # Prioritize exes that match the name
                                for cand in candidates:
                                    if norm_name in cand.stem.lower():
                                        exe = str(cand)
                                        break
                                if not exe:
                                    exe = str(candidates[0])
                        
                        if name:
                            results[_normalize_name(name)] = exe or ""
                except (OSError, ValueError):
                    continue
    except OSError:
        pass
    return results

def scan_start_menu_apps():
    import win32com.client
    try:
        # Use pythoncom for thread safety
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
                            apps[_normalize_name(name)] = target
                    except Exception:
                        pass
    return apps

def _fuzzy_match(query, target):
    """Returns a score for fuzzy matching query in target."""
    q = _normalize_name(query)
    t = _normalize_name(target)
    if q == t: return 1.0
    if q in t: return 0.8 + (len(q) / len(t)) * 0.15
    # Basic word matching
    qw = set(q.split())
    tw = set(t.split())
    if qw.intersection(tw):
        return 0.5 + (len(qw.intersection(tw)) / len(qw)) * 0.3
    return 0.0

def close_app(name):
    target = _normalize_name(name)
    if target in ALIASES:
        target = ALIASES[target]
        
    exe_path = find_app_executable(name)
    target_stems = {target}
    if exe_path:
        target_stems.add(_normalize_name(Path(exe_path).stem))
    
    killed = 0
    try:
        if psutil:
            # Optimization: only iterate once and use a set for lookups
            for proc in psutil.process_iter(["name", "exe"]):
                try:
                    pname = _normalize_name(proc.info.get("name") or "")
                    pstem = _normalize_name(Path(proc.info.get("exe") or pname).stem)
                    if pname in target_stems or pstem in target_stems or any(t in pname for t in target_stems):
                        proc.terminate()
                        killed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        
        if killed:
            msg = f"Closed {name} ({killed} instances)"
            speak(msg)
            status_queue.put(msg)
            return True
            
        # Fallback to taskkill if psutil failed or didn't find it
        if exe_path:
            exe_name = os.path.basename(exe_path)
            subprocess.call(["taskkill", "/IM", exe_name, "/F", "/T"], shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            speak(f"Closed {name}")
            return True
            
    except Exception as e:
        logging.error(f"Error closing app {name}: {e}")
        
    speak(f"Could not find or close {name}")
    return False

def build_app_registry(fast_only=False):
    ensure_appdata_dir()
    idx = {}
    
    # Fast path: Registry is usually enough for installed apps
    idx.update(_scan_uninstall_key(winreg.HKEY_LOCAL_MACHINE))
    idx.update(_scan_uninstall_key(winreg.HKEY_CURRENT_USER))
    
    if not fast_only:
        # Deeper scan for Start Menu and Aliases
        try:
            idx.update(scan_windows_app_aliases())
            idx.update(scan_start_menu_apps())
        except Exception:
            pass

    # Known apps fallback (only if not found)
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
            
    # Persist
    global apps_index
    apps_index.clear()
    apps_index.update(idx)
    _save_apps_index()
    status_queue.put(f"Apps indexed: {len(apps_index)}")

def find_app_executable(name):
    n = _normalize_name(name)
    if n in ALIASES:
        n = ALIASES[n]
        
    if not apps_index:
        _load_apps_index()
    
    # 1. Exact match
    exe = apps_index.get(n)
    if exe and os.path.exists(str(exe).strip(' "\'')):
        return str(exe).strip(' "\'')
        
    # 2. Fuzzy match
    best_match = None
    best_score = 0.65  # Minimum threshold
    
    for k, v in apps_index.items():
        if not v or not os.path.exists(str(v).strip(' "\'')): continue
        score = _fuzzy_match(n, k)
        if score > best_score:
            best_score = score
            best_match = str(v).strip(' "\'')
            
    return best_match

def open_app(name):
    # Try to find without rebuilding first
    exe = find_app_executable(name)
    if not exe:
        # Rebuild registry in a background thread for future use
        threading.Thread(target=build_app_registry, kwargs={"fast_only": False}, daemon=True).start()
        # But for this call, try a quick fast-only rebuild
        build_app_registry(fast_only=True)
        exe = find_app_executable(name)
        
    if exe:
        try:
            if "windowsapps" in exe.lower():
                os.startfile(exe)
            else:
                subprocess.Popen([exe], start_new_session=True)
            status_queue.put(f"Opening {name}")
            speak(f"Opening {name}")
            record_mru(name)
            return True
        except Exception as e:
            logging.error(f"Failed to open {name} at {exe}: {e}")
            status_queue.put(f"Failed to open {name}")
    return False

def install_and_open_app(name):
    status_queue.put(f"Installing {name}...")
    speak(f"Installing {name}")
    def _run():
        try:
            # Attempt winget install
            subprocess.call(["winget", "install", "--silent", name], shell=True)
        except Exception:
            pass
        # Refresh registry and open
        build_app_registry()
        if not open_app(name):
            status_queue.put(f"Could not open {name}")
            speak(f"Could not open {name}")
    threading.Thread(target=_run, daemon=True).start()

def ensure_autostart():
    try:
        import sys
        exe_path = sys.executable if getattr(sys, 'frozen', False) else None
        if exe_path:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run", 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, "SentinelAI", 0, winreg.REG_SZ, exe_path)
    except Exception:
        try:
            subprocess.call(["schtasks", "/Create", "/SC", "ONLOGON", "/TN", "SentinelAI", "/TR", f'"{sys.executable}"'], shell=True)
        except Exception:
            pass

def preflight_bootstrap():
    ensure_appdata_dir()
    _load_apps_index()
    _load_mru()
    settings = _load_settings()
    if settings.get("autostart", True):
        ensure_autostart()
    if not apps_index:
        build_app_registry(fast_only=True)
    threading.Thread(target=build_app_registry, kwargs={"fast_only": False}, daemon=True).start()

def _make_tray_image():
    img = Image.new('RGB', (64, 64), color=(30, 30, 30))
    d = ImageDraw.Draw(img)
    d.ellipse((8,8,56,56), outline=(0,200,255), width=3)
    d.text((18,24), 'SA', fill=(255,255,255))
    return img

def start_tray():
    global tray_icon
    if not _tray_available:
        return
    
    with tray_lock:
        if tray_icon is not None:
            return

        def tray_start(icon, item):
            start_agent()
            return 0
        def tray_stop(icon, item):
            stop_agent()
            return 0
        def tray_toggle_ent(icon, item):
            global entertainment_active
            entertainment_active = not entertainment_active
            speak("Entertainment " + ("enabled" if entertainment_active else "disabled"))
            return 0
        def tray_record(icon, item):
            try:
                status_queue.put("Recording command...")
                try:
                    winsound.Beep(800, 200)
                except Exception:
                    pass
                show_recording_overlay()
                fn, had = record_until_silence(COMMAND_WAV, on_amp=update_recording_overlay)
                close_recording_overlay()
                if not had:
                    status_queue.put("No speech detected.")
                    return 0
                text = transcribe_wav(fn)
                if text:
                    status_queue.put(f"Command: {text}")
                    try:
                        speak(f"You said: {text}")
                    except Exception:
                        pass
                    execute_command(text)
                else:
                    status_queue.put("Transcription empty.")
                    speak("Sorry, I couldn't understand.")
            except Exception:
                pass
            return 0
        def tray_settings(icon, item):
            try:
                open_settings_ui_global()
            except Exception:
                pass
            return 0
        def tray_quit(icon, item):
            icon.stop()
            os._exit(0)
            return 0
        
        menu = pystray.Menu(
            pystray.MenuItem('Start Agent', tray_start),
            pystray.MenuItem('Stop Agent', tray_stop),
            pystray.MenuItem('Record Command', tray_record),
            pystray.MenuItem('Toggle Entertainment', tray_toggle_ent),
            pystray.MenuItem('Settings', tray_settings),
            pystray.MenuItem('Quit', tray_quit)
        )
        tray_icon = pystray.Icon('SentinelAI', _make_tray_image(), 'SentinelAI', menu)
        threading.Thread(target=tray_icon.run, daemon=True).start()

def start_alert_check_loop():
    """Periodically checks for system alerts and reminders."""
    if not sentinel_orchestrator:
        return
    while True:
        try:
            alerts = sentinel_orchestrator.check_periodic_alerts()
            for alert in alerts:
                speak(alert)
                status_queue.put(f"Alert: {alert}")
        except Exception:
            pass
        time.sleep(60)  # check every minute

def start_agent():
    # Turn on the step-by-step helper. It will wait for steps and ask for confirmation.
    global agent_active, pending_step
    agent_active = True
    pending_step = None
    status_queue.put("Agent active. Describe the first step.")
    speak("Agent started. Describe the first step.")
    # Start alert checking
    threading.Thread(target=start_alert_check_loop, daemon=True).start()

def stop_agent():
    # Turn off the step-by-step helper.
    global agent_active, pending_step
    agent_active = False
    pending_step = None
    status_queue.put("Agent stopped.")
    speak("Agent stopped.")

def agent_handle(command):
    # Agent understands basic templates:
    # "install X" → runs winget install X
    # "execute Y" → runs a shell command Y
    # "confirm"   → runs the pending step
    global pending_step
    cmd = (command or "").strip()
    
    # 1. Direct templates
    if cmd.startswith("install "):
        app = cmd.split("install ", 1)[1].strip()
        pending_step = ("winget", ["install", "--silent", app])
        status_queue.put(f"Pending install: {app}. Say 'confirm' to proceed.")
        speak(f"Ready to install {app}. Say confirm to proceed.")
        return
    if cmd.startswith("execute "):
        raw = cmd.split("execute ", 1)[1].strip()
        pending_step = ("cmd", raw)
        status_queue.put("Pending command. Say 'confirm' to run.")
        speak("Pending command. Say confirm to run.")
        return
    if "confirm" in cmd or "yes" in cmd:
        if not pending_step:
            speak("No pending step.")
            return
        kind, payload = pending_step
        try:
            if kind == "winget":
                subprocess.Popen(["winget"] + payload)
            else:
                subprocess.Popen(payload, shell=True)
            status_queue.put("Step executed. Next step?")
            speak("Step executed. What is the next step?")
        except Exception:
            status_queue.put("Step execution failed.")
            speak("Step execution failed.")
        finally:
            pending_step = None
        return

    # 2. Intelligent fallback to Orchestrator or AI
    status_queue.put(f"Agent analyzing step: {cmd}")
    response = None
    
    # Try Orchestrator first
    if sentinel_orchestrator:
        response = sentinel_orchestrator.run_command(cmd)
    
    # If orchestrator didn't return anything, fall back to LLM to generate a command/step
    if not response:
        prompt = f"The user wants to perform this step in an autonomous agent session: '{cmd}'. Provide a concise, actionable response or execute it if you can."
        response = interpret_command(prompt)
        
    if response:
        speak(response)
        status_queue.put("Step processed by AI.")
    else:
        speak("I'm not sure how to perform that step. Please provide a direct command like 'install' or 'execute'.")

def gemini_generate(prompt: str, conv=None, emotion: str = "Neutral", model: str = "gemini-3-flash-preview") -> str:
    """
    Send a prompt to Gemini 2.0 Flash (primary), falling back to OpenRouter or Ollama on failure (e.g. 429 Quota).
    """
    # 1. Try Gemini primary
    key = os.getenv("GEMINI_API_KEY")
    if key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        contents = []
        if conv:
            # Use ConversationManager if provided
            try:
                for turn in list(conv._history):
                    role = "model" if turn.role == "assistant" else "user"
                    contents.append({"role": role, "parts": [{"text": turn.content}]})
            except Exception:
                pass
        
        if not contents:
            contents.append({"role": "user", "parts": [{"text": "You are SentinelAI, an advanced personal AI assistant. Be concise."}]})
        
        user_text = f"[User emotion: {emotion}] {prompt}" if emotion and emotion != "Neutral" else prompt
        contents.append({"role": "user", "parts": [{"text": user_text}]})

        body = {"contents": contents, "generationConfig": {"maxOutputTokens": 400, "temperature": 0.7}}
        try:
            with httpx.Client(timeout=30) as client:
                r = client.post(url, json=body)
                if r.status_code == 429:
                    logging.warning("Gemini 429 Quota Exceeded. Falling back...")
                else:
                    j = r.json()
                    text = j.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                    if text:
                        return text
        except Exception as e:
            logging.warning(f"Gemini call failed: {e}")

    # 2. Fallback to OpenRouter
    or_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_AI_API_KEY")
    if or_key:
        try:
            logging.info("Attempting OpenRouter fallback...")
            # Simple direct request to avoid heavy dependencies in this utility
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {"Authorization": f"Bearer {or_key}", "Content-Type": "application/json"}
            messages = [{"role": "user", "content": prompt}]
            if conv:
                try: messages = conv.build_messages(prompt, emotion=emotion)
                except Exception: pass
            
            payload = {"model": "google/gemini-3-flash-preview", "messages": messages, "max_tokens": 400}
            with httpx.Client(timeout=30) as client:
                r = client.post(url, json=payload, headers=headers)
                if r.status_code != 200:
                    logging.warning(f"OpenRouter returned {r.status_code}: {r.text[:100]}")
                    # Try a different model ID just in case the free experimental one is rotating names
                    if r.status_code == 404:
                        payload["model"] = "google/gemini-3-flash-preview"
                        r = client.post(url, json=payload, headers=headers)
                
                j = r.json()
                text = j.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if text:
                    return text
        except Exception as e:
            logging.warning(f"OpenRouter fallback failed: {e}")

    # 3. Fallback to Ollama
    if sentinel_orchestrator and hasattr(sentinel_orchestrator, 'ollama_module'):
        try:
            logging.info("Attempting local Ollama fallback...")
            return sentinel_orchestrator.ollama_module.generate(prompt) or "Local model returned empty response."
        except Exception as e:
            logging.warning(f"Ollama fallback failed: {e}")

    return "I'm having trouble connecting to all of my intelligence engines. Please check your internet or API keys."

def _cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _mel_filterbank(n_fft, sr, n_mels=40, fmin=300.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(m):
        return 700.0 * (10.0**(m / 2595.0) - 1.0)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fbanks = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]
        if center > left:
            fbanks[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fbanks[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fbanks

def compute_embedding(filepath):
    """
    Computes a speaker embedding from a WAV file.
    Vectorized implementation for maximum speed.
    """
    try:
        x, sr = sf.read(filepath)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x.astype(np.float32)
        
        # Pre-emphasis
        if len(x) > 1:
            x = np.append(x[0], x[1:] - 0.97 * x[:-1])
            
        if len(x) < sr // 2:
            pad = sr // 2 - len(x)
            x = np.pad(x, (0, pad))
            
        frame_len = int(0.025 * sr)
        hop = int(0.010 * sr)
        n_fft = 512 if sr <= 22050 else 1024
        
        # Create frames using stride tricks for efficiency
        num_frames = (len(x) - frame_len) // hop + 1
        if num_frames <= 0:
            return np.zeros(80, dtype=np.float32)
            
        from numpy.lib.stride_tricks import as_strided
        frames = as_strided(x, shape=(num_frames, frame_len), 
                           strides=(x.strides[0] * hop, x.strides[0]))
        
        # Windowing
        window = np.hamming(frame_len)
        frames = frames * window
        
        # RFFT and Power Spectrum
        spec = np.fft.rfft(frames, n=n_fft, axis=1)
        ps = np.abs(spec) ** 2
        
        # Mel filterbank
        fb = _mel_filterbank(n_fft, sr, n_mels=40)
        mel = np.dot(ps[:, :fb.shape[1]], fb.T)
        mel = np.log(mel + 1e-10)
        
        # Stats as embedding (mean and std across time)
        mu = mel.mean(axis=0)
        sigma = mel.std(axis=0)
        emb = np.concatenate([mu, sigma]).astype(np.float32)
        
        # L2 Normalization for more stable cosine similarity
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
            
        return emb
    except Exception as e:
        logging.error(f"Error computing embedding: {e}")
        return np.zeros(80, dtype=np.float32)

# Silero VAD model (loads on first use)
 

# ---------- Utilities ----------

# ── Neural TTS (edge-tts → pyttsx3 fallback) ──────────────────────────────
try:
    from sentinel.voice.tts import get_tts as _get_tts
    _NEURAL_TTS = True
except ImportError:
    _NEURAL_TTS = False

def speak(text, emotion=None, block=False):
    """Non-blocking speech output with neural voice (edge-tts) and emotional tone."""
    if not text:
        return
    text = str(text).strip()
    if not text:
        return
    # Log to status queue for GUI
    try:
        status_queue.put(f"Sentinel: {text[:80]}")
    except Exception:
        pass
    if _NEURAL_TTS:
        _get_tts().speak(text, emotion=emotion or "Neutral", block=block)
    else:
        # Legacy pyttsx3 fallback
        def _s():
            try:
                pythoncom.CoInitialize()
                engine = pyttsx3.init()
                if emotion == "Stressed/Excited":
                    engine.setProperty('rate', engine.getProperty('rate') + 50)
                elif emotion == "Calm/Sad":
                    engine.setProperty('rate', max(engine.getProperty('rate') - 30, 80))
                engine.say(text)
                engine.runAndWait()
                pythoncom.CoUninitialize()
            except Exception as e:
                logging.error(f"pyttsx3 error: {e}")
        if block:
            _s()
        else:
            threading.Thread(target=_s, daemon=True).start()

def show_recording_overlay(parent=None):
    # Create a modern, borderless, always-on-top overlay for audio capture.
    global recording_overlay, recording_canvas, recording_wave_values
    try:
        p = parent or root_window
        win = tk.Toplevel(p) if p else tk.Tk()
        
        # Modern UI styling: borderless, dark, always on top
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.attributes("-alpha", 0.92) # Slight transparency for modern feel
        win.configure(bg="#0f172a") # Dark slate blue
        
        # Center the window on screen
        w, h = 320, 100
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        win.geometry(f"{w}x{h}+{x}+{y}")
        
        # Add a subtle glow/border effect
        frame = tk.Frame(win, bg="#1e293b", bd=1, relief="flat")
        frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        # Label with modern font
        lbl = tk.Label(frame, text="Sentinel Listening...", 
                      fg="#22d3ee", bg="#1e293b", 
                      font=("Segoe UI Variable Display Semibold", 11) if os.name == "nt" else ("Inter", 11))
        lbl.pack(pady=(12, 4))
        
        # Styled waveform canvas
        canvas = tk.Canvas(frame, width=280, height=40, 
                          bg="#0f172a", highlightthickness=1, 
                          highlightbackground="#334155")
        canvas.pack(padx=20, pady=4)
        
        # Make it draggable even without title bar
        def start_move(event):
            win.x = event.x
            win.y = event.y
        def stop_move(event):
            win.x = None
            win.y = None
        def on_move(event):
            deltax = event.x - win.x
            deltay = event.y - win.y
            x = win.winfo_x() + deltax
            y = win.winfo_y() + deltay
            win.geometry(f"+{x}+{y}")
            
        win.bind("<ButtonPress-1>", start_move)
        win.bind("<B1-Motion>", on_move)
        
        recording_overlay = win
        recording_canvas = canvas
        recording_wave_values = []
        
        # Initial draw of static line
        canvas.create_line(0, 20, 280, 20, fill="#334155", width=1, tags="base")
        
        win.update()
    except Exception as e:
        print(f"[Overlay Error] {e}")
        recording_overlay = None
        recording_canvas = None
        recording_wave_values = []

def update_recording_overlay(amp):
    # Draw a scrolling waveform based on recent amplitude values.
    try:
        if recording_canvas is None or recording_overlay is None:
            return
            
        if not recording_overlay.winfo_exists():
            return

        w, h = 280, 40
        mid_y = h // 2
        
        # Normalize amplitude into [0,1] range and keep history
        a = max(0.01, min(1.0, float(amp) * 10.0))
        recording_wave_values.append(a)
        max_points = 50
        if len(recording_wave_values) > max_points:
            recording_wave_values[:] = recording_wave_values[-max_points:]
            
        recording_canvas.delete("wave")
        if len(recording_wave_values) > 1:
            points = []
            spacing = w / max_points
            for i, v in enumerate(recording_wave_values):
                x = int(i * spacing)
                # Draw symmetric bars for a modern "voice" look
                offset = int(v * (h/2.5))
                recording_canvas.create_line(x, mid_y - offset, x, mid_y + offset, 
                                          fill="#22d3ee", width=2, tags="wave", capstyle="round")
        
        recording_overlay.update_idletasks()
        recording_overlay.update()
    except Exception:
        pass

def close_recording_overlay():
    global recording_overlay, recording_overlay_var
    try:
        if recording_overlay is not None:
            recording_overlay.destroy()
    except Exception:
        pass
    recording_overlay = None
    recording_overlay_var = None

# Record a wav using PyAudio for a fixed duration
def record_wav(filename, duration=3, samplerate=16000, channels=1, frames_per_buffer=1024):
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(format=pyaudio.paInt16,
                         channels=channels,
                         rate=samplerate,
                         input=True,
                         frames_per_buffer=frames_per_buffer)
    except Exception as e:
        pa.terminate()
        raise

    frames = []
    num_frames = int(samplerate / frames_per_buffer * duration)
    for _ in range(num_frames):
        data = stream.read(frames_per_buffer, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # write WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(samplerate)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"[Saved] {filename}")
    return filename
def record_until_silence(filename, max_duration=25, samplerate=16000, channels=1, frames_per_buffer=1024, min_duration=1.5, start_timeout=3.5, silence_threshold=0.008, speech_threshold=0.007, silence_duration=0.6, on_amp=None):
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(format=pyaudio.paInt16,
                         channels=channels,
                         rate=samplerate,
                         input=True,
                         frames_per_buffer=frames_per_buffer)
    except Exception:
        pa.terminate()
        raise
    frames = []
    start_t = time.time()
    quiet_t = 0.0
    had_speech = False
    peak_amp = 0.0
    baseline_frames = int(max(1, (samplerate / frames_per_buffer) * 0.5))
    baseline_vals = []
    try:
        while True:
            data = stream.read(frames_per_buffer, exception_on_overflow=False)
            frames.append(data)
            samples = np.frombuffer(data, dtype=np.int16)
            amp = float(np.mean(np.abs(samples))) / 32768.0
            if amp > peak_amp:
                peak_amp = amp
            now = time.time()
            elapsed = now - start_t
            
            if len(baseline_vals) < baseline_frames:
                baseline_vals.append(amp)
                continue
            
            if baseline_vals:
                base = np.median(baseline_vals)
                base = max(base, 0.002)
                sp_thr = max(speech_threshold, base * 2.0)
                si_thr = max(silence_threshold, base * 1.2)
            else:
                sp_thr = speech_threshold
                si_thr = silence_threshold
                
            if amp < si_thr:
                quiet_t += frames_per_buffer / float(samplerate)
            else:
                quiet_t = 0.0
                
            if amp >= sp_thr:
                had_speech = True
                
            if on_amp is not None:
                try:
                    on_amp(amp)
                except Exception:
                    pass
            
            # 1. If we haven't detected speech yet, only stop if we hit start_timeout
            if not had_speech:
                if elapsed >= start_timeout:
                    # Final check of peak amp vs baseline before giving up
                    if peak_amp >= sp_thr * 0.9:
                        had_speech = True
                    else:
                        break # Give up: user never spoke
                continue # Keep listening for the start of speech
                
            # 2. Once speech has started, stop if we've recorded enough (min_duration) 
            # and then detect silence or hit max_duration
            if elapsed >= min_duration:
                if quiet_t >= silence_duration:
                    break
                    
            if elapsed >= max_duration:
                break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(samplerate)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"[Saved] {filename}")
    return filename, had_speech

root_window = None

def shutdown_sentinel():
    """Cleanly deactivates and closes the Sentinel application."""
    global root_window
    print("[Sentinel] Shutdown initiated.")
    try:
        # Stop wake word listener
        stop_wake_listener()
        # Close the main GUI window
        if root_window:
            root_window.after(100, root_window.destroy)
        else:
            import sys
            sys.exit(0)
    except Exception:
        import sys
        sys.exit(0)

def trim_wav_silence(filename, threshold=0.02):
    # This cuts off the quiet parts at the start and end of a sound file
    # so we keep the important speaking part.
    data, sr = sf.read(filename)
    if hasattr(data, "ndim") and data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    amp = np.abs(data)
    idx = np.where(amp > threshold)[0]
    if idx.size == 0:
        return
    pad = int(0.05 * sr)
    start = max(int(idx[0]) - pad, 0)
    end = min(int(idx[-1]) + pad, len(data))
    if end <= start:
        return
    sf.write(filename, data[start:end], sr)

def _audio_signal_stats(filename):
    try:
        data, sr = sf.read(filename)
        if hasattr(data, "ndim") and data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if data.size == 0:
            return {"rms": 0.0, "peak": 0.0, "voiced_ratio": 0.0}
        
        # Avoid division by zero
        amp = np.abs(data)
        rms = float(np.sqrt(np.mean(np.square(data))))
        peak = float(np.max(amp))
        
        # Better voiced ratio calculation
        frame_size = int(sr * 0.02)  # 20ms frames
        energies = []
        for i in range(0, len(amp) - frame_size + 1, frame_size):
            energies.append(float(np.mean(amp[i:i + frame_size])))
        
        if not energies:
            return {"rms": rms, "peak": peak, "voiced_ratio": 0.0}
        
        # Calculate noise floor from the quietest 10% of frames
        sorted_energies = sorted(energies)
        noise_floor = np.mean(sorted_energies[:max(1, len(energies) // 10)])
        
        # Voiced threshold: must be significantly above noise floor AND above a minimal absolute threshold.
        # Further lowered base threshold from 0.005 to 0.003 and multiplier to 1.8 for extreme sensitivity.
        voiced_threshold = max(0.003, min(0.035, noise_floor * 1.8))
        voiced_frames = [e for e in energies if e > voiced_threshold]
        voiced_ratio = float(len(voiced_frames) / len(energies))
        
        return {"rms": rms, "peak": peak, "voiced_ratio": voiced_ratio}
    except Exception:
        return {"rms": 0.0, "peak": 0.0, "voiced_ratio": 0.0}

# Liveness registration / check (Resemblyzer-like embeddings)
def register_reference_if_missing():
    # If we don't have the owner's voice saved yet,
    # we record a short sample and create its fingerprint.
    if not os.path.exists(OWNER_EMBED_PATH):
        # Prompt for master password before registration
        conn = _ensure_secrets_db()
        master = _prompt_master(root_window)
        if not master:
            speak("Master password required to register voice.")
            status_queue.put("Voice registration aborted: no password.")
            conn.close()
            return False
        if not _verify_master(conn, master):
            speak("Invalid master password.")
            status_queue.put("Voice registration failed: wrong password.")
            conn.close()
            return False
        conn.close()

        speak("No registered voice found. Please say the passphrase after the beep.", block=True)
        show_recording_overlay(root_window)
        try:
            winsound.Beep(800, 200)
        except Exception:
            pass
            
        print("[Recording reference voice]")
        _, had = record_until_silence(REFERENCE_WAV, max_duration=5, min_duration=2.0, on_amp=update_recording_overlay)
        close_recording_overlay()
        if not had:
            speak("Voice registration failed. No speech detected.")
            status_queue.put("Voice registration failed.")
            return False
        trim_wav_silence(REFERENCE_WAV)
        stats = _audio_signal_stats(REFERENCE_WAV)
        if stats["rms"] < 0.005 or stats["voiced_ratio"] < 0.10:
            speak("Voice registration failed. Please speak clearly, possibly closer to the microphone.")
            status_queue.put(f"Voice registration too quiet (RMS: {stats['rms']:.3f}) or unclear (VR: {stats['voiced_ratio']:.2f}).")
            return False
        owner_embed = compute_embedding(REFERENCE_WAV)
        np.save(OWNER_EMBED_PATH, owner_embed)
        speak("Voice registered successfully.")
        status_queue.put("Voice registered.")
        return True
    return True

def manual_register_voice():
    # Button in the screen: lets you re-record the owner's voice.
    """Triggered by GUI button: re-record owner reference voice."""
    try:
        # Prompt for master password before update
        conn = _ensure_secrets_db()
        master = _prompt_master(root_window)
        if not master:
            speak("Master password required to update voice.")
            status_queue.put("Voice update aborted: no password.")
            conn.close()
            return
        if not _verify_master(conn, master):
            speak("Invalid master password.")
            status_queue.put("Voice update failed: wrong password.")
            conn.close()
            return
        conn.close()

        speak("Please say your reference passphrase after the beep.", block=True)
        show_recording_overlay(root_window)
        try:
            winsound.Beep(800, 200)
        except Exception:
            pass

        status_queue.put("Recording new reference voice...")
        print("[Manual registration] Recording new reference...")

        _, had = record_until_silence(REFERENCE_WAV, max_duration=6, min_duration=2.5, on_amp=update_recording_overlay)
        close_recording_overlay()

        if not had:
            speak("Voice update failed. No speech detected.")
            status_queue.put("Voice update failed: no speech.")
            return
            
        stats = _audio_signal_stats(REFERENCE_WAV)
        if stats["rms"] < 0.005 or stats["voiced_ratio"] < 0.10:
            speak("Voice update failed. Audio quality too low or too quiet.")
            status_queue.put(f"Voice update failed: poor audio (RMS: {stats['rms']:.3f}, VR: {stats['voiced_ratio']:.2f}).")
            return

        owner_embed = compute_embedding(REFERENCE_WAV)
        np.save(OWNER_EMBED_PATH, owner_embed)

        speak("Voice registration updated successfully.")
        status_queue.put("Voice registration updated successfully.")

        print("[Manual registration] Updated owner_embed.npy")
    except Exception as e:
        speak("Voice registration failed.")
        status_queue.put(f"Registration error: {e}")
        print("[Manual registration error]", e)

def liveness_check(threshold=0.88):
    # Checks quickly if the new voice sample looks like the owner's
    # by comparing their fingerprints.
    # Ensure reference exists
    if not os.path.exists(OWNER_EMBED_PATH):
        if not register_reference_if_missing():
            return False

    # Block while speaking so we don't start recording user voice (or computer voice) too early
    speak("Please repeat the passphrase after the beep.", block=True)
    
    show_recording_overlay(root_window)
    try:
        winsound.Beep(800, 200)
    except Exception:
        pass
        
    print("[Recording liveness sample]")
    # Use a longer start_timeout to give the user time to react
    _, had = record_until_silence(LIVENESS_WAV, max_duration=5, min_duration=1.2, start_timeout=4.0, silence_duration=0.45, on_amp=update_recording_overlay)
    close_recording_overlay()
    if not had:
        status_queue.put("Liveness failed: no speech.")
        return False
    try:
        trim_wav_silence(LIVENESS_WAV)
        stats = _audio_signal_stats(LIVENESS_WAV)
        # Even more relaxed checks for liveness
        if stats["rms"] < 0.005 or stats["peak"] < 0.02 or stats["voiced_ratio"] < 0.12:
            status_queue.put(f"Liveness failed: weak audio (RMS: {stats['rms']:.3f}, VR: {stats['voiced_ratio']:.2f})")
            return False
            
        live_embed = compute_embedding(LIVENESS_WAV)
        owner_embed = np.load(OWNER_EMBED_PATH)
        similarity = _cosine(live_embed, owner_embed)
        print(f"[Liveness Similarity Score]: {similarity:.4f}")
        status_queue.put(f"Liveness score: {similarity:.3f}")
        
        # Extremely relaxed similarity threshold and voiced ratio requirement
        is_owner = similarity >= threshold or (similarity >= 0.85 and stats["voiced_ratio"] >= 0.15)
        if not is_owner:
            status_queue.put("Voice verification failed.")
            speak("Voice verification failed. Access denied.")
        return is_owner
    except Exception as e:
        print("[Liveness error]", e)
        return False

# Transcribe a WAV file using SpeechRecognition (Google)
def _has_network():
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except Exception:
        return False

def transcribe_wav_offline(filename):
    try:
        import vosk  # local import to avoid hard dependency
    except Exception:
        return ""
    model_dir = os.getenv("VOSK_MODEL") or os.path.join(APPDATA_DIR, "vosk-model")
    if not os.path.isdir(model_dir):
        s = _load_settings()
        if bool(s.get("offline_stt", True)) and ensure_vosk_model():
            pass
        else:
            return ""
    try:
        model = vosk.Model(model_dir)
        rec = vosk.KaldiRecognizer(model, 16000)
        wf = wave.open(filename, "rb")
        try:
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)
        finally:
            wf.close()
        import json as _json
        res = _json.loads(rec.Result())
        text = (res.get("text") or "").strip()
        return text.lower()
    except Exception:
        return ""
def transcribe_wav(filename):
    """
    Transcribe a WAV file to text.
    Uses faster-whisper (local) → Google STT → Vosk as fallback chain.
    """
    try:
        from sentinel.voice.stt import transcribe as _stt_transcribe
        result = _stt_transcribe(filename)
        if result:
            logging.info(f"[STT] {result}")
            return result
    except Exception as e:
        logging.warning(f"New STT module failed: {e}, falling back to Google STT")

    # Legacy Google STT fallback
    try:
        r = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        logging.info(f"[Google STT] {text}")
        return text.lower()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return transcribe_wav_offline(filename)

# Interpret command — context-aware, multi-turn, Gemini 2.0
def interpret_command(command: str, emotion: str = "Neutral") -> str:
    """
    Routes a command to the best available LLM with full conversation history.
    Now wraps the unified gemini_generate which handles fallbacks.
    """
    if not command:
        return ""

    # Load conversation manager
    try:
        from sentinel.core.conversation import get_conversation
        conv = get_conversation()
    except Exception:
        conv = None

    # Unified call via gemini_generate which handles fallback chain: Gemini -> OpenRouter -> Ollama
    resp = gemini_generate(command, conv=conv, emotion=emotion)
    
    if resp and conv:
        conv.add_turn("user", command, emotion=emotion)
        conv.add_turn("assistant", resp)
        
    return resp

# Execute a handful of commands (keeps your original behaviors)
def execute_command(command):
    # Main entry point for command routing with Phase 3 Error Repair.
    try:
        _execute_command_internal(command)
    except Exception as e:
        logging.error(f"Execution crash: {e}")
        # Phase 3: Autonomous Error Self-Repair
        if sentinel_orchestrator:
            status_queue.put("System error detected. Initiating self-repair...")
            repair_prompt = f"The user tried to execute: '{command}', but it failed with error: '{e}'. Suggest a corrected command or a different way to achieve the goal."
            repair_suggestion = interpret_command(repair_prompt)
            if repair_suggestion:
                speak(f"I encountered an error, but I have a repair suggestion: {repair_suggestion}. Should I try that?")
            else:
                speak("I encountered a system error and could not find an immediate repair path.")

def _execute_command_internal(command):
    # Modular command engine entry point.
    original = (command or "").strip('.!?, ')
    command = original.lower()
    print(f"[Sentinel] Processing: {command}")
    
    # ─── Priority System Overrides ───────────────────────────────────────
    if command in ("stop", "stop agent", "agent stop", "disable agent"):
        speak("Stopping agent")
        stop_agent()
        return
    if command in ("stop listening", "pause listening", "do not listen"):
        global listening_blocked_until
        listening_blocked_until = time.time() + 300
        speak("Pausing listening for five minutes")
        return
    if command in ("start listening", "resume listening"):
        listening_blocked_until = 0
        speak("Listening resumed")
        return
    if command in ("enable entertainment", "entertain me", "talk mode"):
        global entertainment_active
        entertainment_active = True
        status_queue.put("Entertainment mode enabled.")
        speak("Entertainment mode enabled.")
        return
    if command in ("disable entertainment", "quiet mode", "stop talking"):
        entertainment_active = False
        status_queue.put("Entertainment mode disabled.")
        speak("Entertainment mode disabled.")
        return
    
    # ─── Modular Routing ────────────────────────────────────────────────
    if sentinel_orchestrator:
        status_queue.put("Routing command...")
        response = sentinel_orchestrator.run_command(original)
        if response:
            speak(response)
            return


    # ─── Last Resort Fallback ───────────────────────────────────────────
    ai_keywords = ["ai", "gemini", "think", "explain", "why", "how", "what is", "who is"]
    if any(k in command for k in ai_keywords) or "?" in command:
        status_queue.put("Consulting Gemini...")
        response = interpret_command(original)
        if response:
            speak(response)
            return

    speak("I'm sorry, I couldn't understand or execute that command.")


# ---------- Wake-word listener (runs in background thread) ----------
def listen_for_wake_word_loop(session_duration=3600):
    # Background loop:
    global next_agent_capture_time, porcupine_status
    # 1) Wait for wake word (or timed agent capture)
    # 2) Check voice liveness (if needed)
    # 3) Record short command and execute it
    """Listens for Porcupine wake word and then handles auth + command."""
    session_valid_until = 0
    sens = 0.6
    kw = ["hey computer"]
    if not ACCESS_KEY:
        status_queue.put("Porcupine key missing. Wake word disabled.")
        porcupine_status = "Missing Key"
        return
    try:
        try:
            s = _load_settings()
            sens = float(s.get("wake_sensitivity", 0.6))
        except Exception:
            pass
        try:
            s = _load_settings()
            raw = s.get("wake_keywords") or s.get("wake_keyword")
            if raw:
                kw = [k.strip() for k in str(raw).split(',') if k.strip()]
        except Exception:
            pass

        porcupine = None
        model_path = _porcupine_model_path()
        if os.path.exists(PORCUPINE_KEYWORD_PATH):
            print(f"[Porcupine] Found keyword file: {PORCUPINE_KEYWORD_PATH}")
            try:
                if model_path:
                    porcupine = pvporcupine.create(
                        access_key=ACCESS_KEY,
                        keyword_paths=[PORCUPINE_KEYWORD_PATH],
                        sensitivities=[sens],
                        model_path=model_path
                    )
                else:
                    porcupine = pvporcupine.create(
                        access_key=ACCESS_KEY,
                        keyword_paths=[PORCUPINE_KEYWORD_PATH],
                        sensitivities=[sens]
                    )
                print(f"[Porcupine] Loaded custom keyword file successfully.")
            except Exception as e:
                print(f"[Porcupine] Failed to load custom keyword file: {e}")
                porcupine = None
        
        if porcupine is None:
            print(f"[Porcupine] Using default keywords: {kw}")
            print(f"[Porcupine] Sensitivity: {sens}")
            if model_path:
                porcupine = pvporcupine.create(
                    access_key=ACCESS_KEY,
                    keywords=kw,
                    sensitivities=[sens] if len(kw) <= 1 else [sens] * len(kw),
                    model_path=model_path
                )
            else:
                porcupine = pvporcupine.create(
                    access_key=ACCESS_KEY,
                    keywords=kw,
                    sensitivities=[sens] if len(kw) <= 1 else [sens] * len(kw)
                )
        
        porcupine_status = "OK"
    except Exception as e:
        err_msg = str(e)
        print("[Porcupine init error]", err_msg)
        if "Invalid access_key" in err_msg:
            status_queue.put("Porcupine key invalid.")
            porcupine_status = "Invalid Key"
        elif "Version mismatch" in err_msg:
            status_queue.put("Porcupine version mismatch.")
            porcupine_status = "Version Error"
        else:
            fallback_ok = False
            if "keyword" in err_msg.lower() or "model" in err_msg.lower():
                try:
                    if model_path:
                        porcupine = pvporcupine.create(
                            access_key=ACCESS_KEY,
                            keywords=["computer"],
                            sensitivities=[sens],
                            model_path=model_path
                        )
                    else:
                        porcupine = pvporcupine.create(
                            access_key=ACCESS_KEY,
                            keywords=["computer"],
                            sensitivities=[sens]
                        )
                    porcupine_status = "OK"
                    status_queue.put("Porcupine fallback: computer")
                    fallback_ok = True
                except Exception as e2:
                    err_msg = str(e2)
            if not fallback_ok:
                status_queue.put(f"Porcupine failed: {err_msg[:20]}")
                porcupine_status = "Failed"
                return

    pa = pyaudio.PyAudio()  # open the microphone for reading
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=porcupine.sample_rate,
                     input=True,
                     frames_per_buffer=porcupine.frame_length)

    status_queue.put("Listening for wake word...")
    print("[Porcupine running]")

    try:
        while not wake_stop_event.is_set():
            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm_unpacked = np.frombuffer(pcm, dtype=np.int16)
            keyword_index = porcupine.process(pcm_unpacked)
            now_t = time.time()
            should_capture = keyword_index >= 0 or (agent_active and now_t >= next_agent_capture_time)
            if should_capture:
                print("[Wake word detected]")
                status_queue.put("Wake word detected.")
                
                # Use blocking speech so the "Ready" greeting finishes before we start listening
                if not agent_active:
                    speak("Yes Master", block=True)
                    
                now = time.time()
                if now > session_valid_until and not agent_active:
                    status_queue.put("Performing liveness check...")
                    if not liveness_check():
                        # Access denied speech should probably be non-blocking to allow the loop to continue
                        speak("Access denied. Voice does not match.")
                        status_queue.put("Liveness failed.")
                        continue
                    else:
                        session_valid_until = time.time() + session_duration
                        status_queue.put("Liveness passed. Session active.")

                # respect listening pause
                if time.time() < listening_blocked_until:
                    continue
                    
                # Record user command
                status_queue.put("Recording command...")
                show_recording_overlay(root_window)
                try:
                    winsound.Beep(800, 200)
                except Exception:
                    pass
                    
                # Dynamic recording: 5s min start timeout, until user stops speaking
                fn, had = record_until_silence(COMMAND_WAV, max_duration=12, min_duration=0.9, start_timeout=5.0, silence_duration=0.5, on_amp=update_recording_overlay)
                close_recording_overlay()
                if not had:
                    status_queue.put("No speech detected.")
                    continue
                status_queue.put("Processing command...")

                # transcribe & execute
                cmd_text = transcribe_wav(fn)
                if not cmd_text:
                    speak("Sorry, I couldn't understand the command.")
                    status_queue.put("Transcription empty.")
                else:
                    # Emotion Analysis
                    emotion = "Neutral"
                    try:
                        from sentinel.modules.emotion import EmotionModule
                        emotion_module = EmotionModule()
                        emotion = emotion_module.analyze_audio_emotion(fn)
                        suggestion = emotion_module.suggest_action_based_on_emotion(emotion)
                        if suggestion:
                            speak(suggestion, emotion=emotion)
                            status_queue.put(f"Emotion detected: {emotion}")
                    except Exception:
                        pass

                    status_queue.put(f"Command: {cmd_text}")
                    try:
                        speak(f"You said: {cmd_text}", emotion=emotion)
                    except Exception:
                        pass
                    execute_command(cmd_text)

                if agent_active:
                    next_agent_capture_time = time.time() + 6  # listen again soon (in 6s)

    except Exception as e:
        print("[Wake loop error]", e)
        status_queue.put("Error in wake listener.")
    finally:
        try:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            porcupine.delete()
        except Exception:
            pass

def start_wake_listener():
    global wake_thread
    if wake_thread and wake_thread.is_alive():
        return
    wake_stop_event.clear()
    wake_thread = threading.Thread(target=listen_for_wake_word_loop, daemon=True)
    wake_thread.start()

def stop_wake_listener():
    try:
        wake_stop_event.set()
    except Exception:
        pass

def restart_wake_listener():
    stop_wake_listener()
    try:
        if wake_thread:
            wake_thread.join(timeout=2)
    except Exception:
        pass
    start_wake_listener()

def refresh_config():
    """Refreshes configuration from environment variables and restarts services."""
    global ACCESS_KEY, sentinel_orchestrator, wake_thread
    
    # Reload from AppData .env to be sure
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
        
    ACCESS_KEY = os.getenv("PVPORCUPINE_PRIVATE_KEY")
    
    # Initialize/Refresh orchestrator
    if os.getenv("GEMINI_API_KEY"):
        try:
            from sentinel.core.orchestrator import SentinelOrchestrator
            if sentinel_orchestrator is None:
                sentinel_orchestrator = SentinelOrchestrator(
                    llm_callback=lambda p: gemini_generate(p),
                    speak_fn=speak,
                    exit_callback=shutdown_sentinel
                )
        except Exception:
            pass

    # Restart Wake Word Listener if it's not running
    if ACCESS_KEY:
        start_wake_listener()

def update_mic_level():
    """Continuously updates the mic level variable for the GUI."""
    global mic_level_var
    if mic_level_var is None:
        return

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=512)
        while True:
            try:
                data = stream.read(512, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16)
                amp = float(np.mean(np.abs(samples))) / 32768.0
                # Ensure mic_level_var is still valid and initialized
                if isinstance(mic_level_var, tk.DoubleVar):
                    try:
                        mic_level_var.set(amp)
                    except (tk.TclError, AttributeError):
                        pass
            except Exception:
                if mic_level_var:
                    mic_level_var.set(0.0)
                break
            time.sleep(0.05)
    except Exception:
        if mic_level_var:
            mic_level_var.set(0.0)
    finally:
        try:
            stream.close()
            pa.terminate()
        except Exception:
            pass

# ---------- GUI ----------
# def start_gui():

#     root = tk.Tk()
#     root.title("SentinelAI Dashboard")
#     root.geometry("380x180")

#     status_label = tk.Label(root, text="SentinelAI Running...", font=("Arial", 11))
#     status_label.pack(pady=8)

#     session_label = tk.Label(root, text="Session: expired", font=("Arial", 10))
#     session_label.pack()

#     diag_items = []
#     diag_items.append("OpenAI: OK" if os.getenv("OPEN_AI_API_KEY") else "OpenAI: Missing")
#     diag_items.append("Porcupine: OK" if os.getenv("PVPORCUPINE_PRIVATE_KEY") else "Porcupine: Missing")
#     diag_items.append("Driver: OK" if os.path.exists(os.path.join(os.path.dirname(__file__), "chromedriver-win64", "chromedriver.exe")) else "Driver: Missing")
#     diag_label = tk.Label(root, text=" | ".join(diag_items), font=("Arial", 9))
#     diag_label.pack(pady=6)

#     def update_ui():
#         # drain status queue
#         while not status_queue.empty():
#             msg = status_queue.get_nowait()
#             status_label.config(text=msg)
#         root.after(800, update_ui)

#     update_ui()
#     root.mainloop()

# ── Cyberpunk Glassmorphism UI Constants ─────────────────────────────────
MODERN_BG = "#020617"         # Deepest Midnight
MODERN_SURFACE = "#0f172a"    # Slate 900
MODERN_ACCENT = "#06b6d4"     # Cyan 500
MODERN_GLOW = "#0891b2"       # Cyan 600
MODERN_TEXT = "#f8fafc"       # Slate 50
MODERN_TEXT_MUTED = "#64748b" # Slate 500
MODERN_SUCCESS = "#10b981"    # Emerald 500
MODERN_WARNING = "#f59e0b"    # Amber 500
MODERN_DANGER = "#ef4444"     # Rose 500
MODERN_FONT = ("Consolas", 10)
MODERN_FONT_BOLD = ("Consolas", 10, "bold")
MODERN_FONT_LARGE = ("Consolas", 16, "bold")

def apply_modern_styles(root):
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass

    root.configure(bg=MODERN_BG)
    root.attributes("-alpha", 0.98) # Slight window transparency

    # Custom TFrame for glass effect
    style.configure("Glass.TFrame", background=MODERN_SURFACE, relief="flat")
    
    # Label Styles
    style.configure("Modern.TLabel", background=MODERN_BG, foreground=MODERN_TEXT, font=MODERN_FONT)
    style.configure("ModernMuted.TLabel", background=MODERN_BG, foreground=MODERN_TEXT_MUTED, font=MODERN_FONT)
    style.configure("ModernHeader.TLabel", background=MODERN_BG, foreground=MODERN_ACCENT, font=MODERN_FONT_LARGE)
    
    # Surface Label Styles (for items inside frames)
    style.configure("Surface.TLabel", background=MODERN_SURFACE, foreground=MODERN_TEXT, font=MODERN_FONT)
    style.configure("SurfaceMuted.TLabel", background=MODERN_SURFACE, foreground=MODERN_TEXT_MUTED, font=MODERN_FONT)

    # Button Styles - Cyberpunk Neon Look
    style.configure("Modern.TButton", 
                   padding=(12, 6), 
                   relief="flat", 
                   background="#1e293b", 
                   foreground=MODERN_TEXT,
                   font=MODERN_FONT_BOLD,
                   borderwidth=1)
    style.map("Modern.TButton",
              background=[('active', MODERN_ACCENT), ('pressed', MODERN_SURFACE)],
              foreground=[('active', MODERN_BG)],
              bordercolor=[('active', MODERN_ACCENT)])

    style.configure("ModernAccent.TButton", 
                   padding=(12, 6), 
                   relief="flat", 
                   background=MODERN_ACCENT, 
                   foreground=MODERN_BG,
                   font=MODERN_FONT_BOLD)
    style.map("ModernAccent.TButton",
              background=[('active', MODERN_TEXT), ('pressed', MODERN_ACCENT)])

    # Progressbar - Cyber Glow
    style.configure("Modern.Horizontal.TProgressbar", 
                   troughcolor="#020617", 
                   background=MODERN_ACCENT, 
                   thickness=6,
                   borderwidth=0)

    # Combobox - Sleek Dark
    style.configure("Modern.TCombobox", 
                   fieldbackground=MODERN_SURFACE, 
                   background=MODERN_SURFACE, 
                   foreground=MODERN_TEXT,
                   arrowcolor=MODERN_ACCENT,
                   font=MODERN_FONT)

def start_gui():
    # Builds a cyberpunk, glassmorphism dashboard.
    root = tk.Tk()
    global root_window, mic_level_var
    root_window = root
    
    apply_modern_styles(root)
    
    root.title("SENTINEL_OS_V4")
    root.geometry("540x740")
    root.resizable(True, True)
    root.minsize(500, 600) # Ensure it doesn't get too small

    # Main container with deep space background
    main_frame = tk.Frame(root, bg=MODERN_BG, padx=25, pady=25)
    main_frame.pack(fill='both', expand=True)

    # Header with Neon Glow
    header_frame = tk.Frame(main_frame, bg=MODERN_BG)
    header_frame.pack(fill='x', pady=(0, 25))
    
    title_label = tk.Label(header_frame, text="SENTINEL_CORE_V4", font=MODERN_FONT_LARGE, 
                          bg=MODERN_BG, fg=MODERN_ACCENT)
    title_label.pack(side='left')
    
    status_label = tk.Label(header_frame, text="SYSTEM_LINK_ACTIVE", 
                           font=MODERN_FONT, bg=MODERN_BG, fg=MODERN_TEXT_MUTED)
    status_label.pack(side='right')

    # Healthcare / Status line (Brain & Local)
    health_frame = tk.Frame(main_frame, bg=MODERN_BG)
    health_frame.pack(fill='x', pady=(0, 15))
    
    brain_label = tk.Label(health_frame, text="BRAIN: CONNECTING", 
                          bg=MODERN_BG, fg="#94a3b8", font=("Consolas", 8))
    brain_label.pack(side='left')
    
    local_label = tk.Label(health_frame, text="LOCAL: INIT", 
                          bg=MODERN_BG, fg="#94a3b8", font=("Consolas", 8))
    local_label.pack(side='right')

    # Data Stream Monitor (Glassmorphism effect)
    info_frame = ttk.Frame(main_frame, style="Glass.TFrame")
    info_frame.pack(fill='x', pady=(0, 25))
    
    info_content = tk.Frame(info_frame, bg=MODERN_SURFACE, padx=20, pady=20)
    info_content.pack(fill='both', expand=True)

    session_label = tk.Label(info_content, text="[SECURE_SESSION]: EXPIRED", 
                            font=MODERN_FONT, bg=MODERN_SURFACE, fg=MODERN_TEXT)
    session_label.pack(anchor='w')

    # System Diagnostics Grid
    diag_items = []
    diag_items.append("NET: OK" if (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_AI_API_KEY")) else "NET: ERR")
    diag_items.append(f"VOICE: {porcupine_status}")
    diag_items.append("LLM: OK" if os.getenv("GEMINI_API_KEY") else "LLM: ERR")
    
    diag_label = tk.Label(info_content, text=" | ".join(diag_items), 
                         font=("Consolas", 8), 
                         bg=MODERN_SURFACE, fg=MODERN_ACCENT)
    diag_label.pack(anchor='w', pady=(12, 0))

    # Neural Input (Mic Bar)
    mic_frame = tk.Frame(main_frame, bg=MODERN_BG)
    mic_frame.pack(fill='x', pady=(0, 25))
    
    tk.Label(mic_frame, text="NEURAL_INPUT_ACTIVE", font=MODERN_FONT_BOLD, 
             bg=MODERN_BG, fg=MODERN_TEXT).pack(anchor='w', pady=(0, 8))
    
    mic_level_var = tk.DoubleVar(value=0.0)
    mic_bar = ttk.Progressbar(mic_frame, orient='horizontal', mode='determinate', 
                             maximum=0.2, variable=mic_level_var, style="Modern.Horizontal.TProgressbar")
    mic_bar.pack(fill='x')

    threading.Thread(target=update_mic_level, daemon=True).start()

    # Tactical Operations Grid
    actions_frame = tk.Frame(main_frame, bg=MODERN_BG)
    actions_frame.pack(fill='x', pady=(0, 25))
    actions_frame.columnconfigure((0, 1), weight=1, pad=12)

    ttk.Button(actions_frame, text="REGISTER_VOICE", style="Modern.TButton", 
               command=manual_register_voice).grid(row=0, column=0, sticky='ew', pady=6)
    
    def refresh_apps():
        status_queue.put("Refreshing apps registry...")
        threading.Thread(target=build_app_registry, daemon=True).start()

    ttk.Button(actions_frame, text="REBUILD_INDEX", style="Modern.TButton", 
               command=refresh_apps).grid(row=0, column=1, sticky='ew', pady=6)

    ttk.Button(actions_frame, text="USAGE_STATS", style="Modern.TButton", 
               command=lambda: messagebox.showinfo("Top Usage", "\n".join(top_mru(5)) or "No data")).grid(row=1, column=0, sticky='ew', pady=6)
    
    ttk.Button(actions_frame, text="MACRO_SEQUENCER", style="Modern.TButton", 
               command=lambda: open_macros_ui(root)).grid(row=1, column=1, sticky='ew', pady=6)

    # Deployment Matrix (Launcher)
    launcher_frame = ttk.Frame(main_frame, style="Glass.TFrame")
    launcher_frame.pack(fill='x', pady=(0, 25))
    
    launcher_content = tk.Frame(launcher_frame, bg=MODERN_SURFACE, padx=20, pady=20)
    launcher_content.pack(fill='both', expand=True)

    tk.Label(launcher_content, text="DEPLOYMENT_MATRIX", font=MODERN_FONT_BOLD, 
             bg=MODERN_SURFACE, fg=MODERN_ACCENT).pack(anchor='w', pady=(0, 12))

    apps_var = tk.StringVar()
    mru_items = top_mru(10)
    mru_combo = ttk.Combobox(launcher_content, textvariable=apps_var, values=mru_items, 
                            state='readonly', style="Modern.TCombobox")
    mru_combo.set(mru_items[0] if mru_items else '')
    mru_combo.pack(fill='x', pady=(0, 12))

    btn_row = tk.Frame(launcher_content, bg=MODERN_SURFACE)
    btn_row.pack(fill='x')
    
    ttk.Button(btn_row, text="EXECUTE_LAUNCH", style="ModernAccent.TButton", 
               command=lambda: open_app(apps_var.get()) if apps_var.get() else None).pack(side='left', expand=True, fill='x', padx=(0, 8))

    ttk.Button(btn_row, text="RFRSH", width=6, style="Modern.TButton", 
               command=lambda: mru_combo.configure(values=top_mru(10))).pack(side='right')

    # System Override
    footer_frame = tk.Frame(main_frame, bg=MODERN_BG)
    footer_frame.pack(fill='x', side='bottom')

    def manual_run_command():
        status_queue.put("Recording command...")
        show_recording_overlay(root)
        fn, had = record_until_silence(COMMAND_WAV, on_amp=update_recording_overlay)
        close_recording_overlay()
        if had:
            text = transcribe_wav(fn)
            if text: execute_command(text)

    ttk.Button(footer_frame, text="[ INITIATE_MANUAL_OVERRIDE ]", style="ModernAccent.TButton", 
               command=manual_run_command).pack(fill='x', pady=(0, 12))

    ctrl_row = tk.Frame(footer_frame, bg=MODERN_BG)
    ctrl_row.pack(fill='x')
    ctrl_row.columnconfigure((0, 1, 2), weight=1, pad=8)

    ttk.Button(ctrl_row, text="CONFIG", style="Modern.TButton", 
               command=lambda: open_settings_ui_global(root)).grid(row=0, column=0, sticky='ew')
    
    ttk.Button(ctrl_row, text="VAULT", style="Modern.TButton", 
               command=lambda: ensure_env_setup(force=True)).grid(row=0, column=1, sticky='ew')

    ttk.Button(ctrl_row, text="TERMINATE", style="Modern.TButton", 
               command=root.destroy).grid(row=0, column=2, sticky='ew')

    agent_row = tk.Frame(footer_frame, bg=MODERN_BG)
    agent_row.pack(fill='x', pady=(12, 0))
    agent_row.columnconfigure((0, 1), weight=1, pad=8)

    ttk.Button(agent_row, text="INITIALIZE_AGENT", style="Modern.TButton", 
               command=start_agent).grid(row=0, column=0, sticky='ew')
    
    ttk.Button(agent_row, text="DISABLE_AGENT", style="Modern.TButton", 
               command=stop_agent).grid(row=0, column=1, sticky='ew')

    def on_close():
        try:
            start_tray()
            root.withdraw()
            status_queue.put("System minimized to tray.")
        except Exception:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    def update_ui():
        # Update command status
        while not status_queue.empty():
            msg = status_queue.get_nowait()
            status_label.config(text=msg.upper().replace(" ", "_"))
            if any(x in msg for x in ["Wake", "Processing", "Thinking"]):
                status_label.config(fg=MODERN_ACCENT)
            else:
                status_label.config(fg=MODERN_TEXT_MUTED)
        
        # Update Brain/Local health status
        brain_label.config(text=f"BRAIN: {brain_status}", fg=brain_status_color)
        local_label.config(text=f"LOCAL: {local_status}", fg=local_status_color)
        
        root.after(400, update_ui)

    update_ui()
    root.mainloop()


# ---------- Main ----------
def main():
    # Program start:
    ensure_appdata_dir()
    if "--setup-wizard" in sys.argv:
        ensure_env_setup(force=True)
        return
    if not ensure_env_setup():
        try:
            subprocess.Popen([sys.executable, "--setup-wizard"])
        except Exception:
            return
        return
    refresh_config()  # Load configuration and initialize orchestrator
    # 1) Make sure we have the owner's voice saved
    # 2) Start listening in the background
    # 3) Show the small dashboard
    # Ensure owner reference exists (register if missing)
    register_reference_if_missing()
    preflight_bootstrap()

    # Start listener thread
    start_wake_listener()

    # Start GUI (blocks main thread) unless background mode
    bg = (os.getenv("SENTINEL_BACKGROUND","0").lower() in ("1","true","yes"))
    if not bg:
        start_gui()
    else:
        start_tray()

pass
def open_settings_ui_global(parent=None):
    s = _load_settings()
    win = tk.Toplevel(parent) if parent else tk.Tk()
    win.title("Sentinel Settings")
    win.geometry("400x600")
    win.configure(bg=MODERN_BG)
    
    # Use modern styles if already defined in start_gui
    style = ttk.Style(win)
    
    main_frame = tk.Frame(win, bg=MODERN_BG, padx=20, pady=20)
    main_frame.pack(fill='both', expand=True)

    tk.Label(main_frame, text="SETTINGS", font=MODERN_FONT_LARGE, 
             bg=MODERN_BG, fg=MODERN_ACCENT).pack(anchor='w', pady=(0, 20))

    ent_var = tk.BooleanVar(value=entertainment_active)
    auto_var = tk.BooleanVar(value=bool(s.get("autostart", True)))
    bg_var = tk.BooleanVar(value=bool(s.get("background", False)))
    sens_var = tk.DoubleVar(value=float(s.get("wake_sensitivity", 0.6)))
    kw_var = tk.StringVar(value=str(s.get("wake_keywords", s.get("wake_keyword", "hey computer"))))
    offline_var = tk.BooleanVar(value=bool(s.get("offline_stt", True)))
    strong_vault_var = tk.BooleanVar(value=bool(s.get("require_strong_vault", False)))

    # Modernized Checkbuttons
    def create_check(text, var):
        cb = tk.Checkbutton(main_frame, text=text, variable=var, 
                           bg=MODERN_BG, fg=MODERN_TEXT, 
                           activebackground=MODERN_BG, activeforeground=MODERN_ACCENT,
                           selectcolor=MODERN_SURFACE, font=MODERN_FONT,
                           padx=10, pady=5, anchor='w')
        cb.pack(fill='x')

    create_check("Entertainment Mode", ent_var)
    create_check("Autostart on Login", auto_var)
    create_check("Run in Background", bg_var)
    create_check("Enable Offline STT", offline_var)
    create_check("Strong Vault (AES-GCM)", strong_vault_var)

    # Sensitivity Scale
    tk.Label(main_frame, text="Wake Sensitivity", font=MODERN_FONT_BOLD, 
             bg=MODERN_BG, fg=MODERN_TEXT).pack(anchor='w', pady=(15, 5))
    ttk.Scale(main_frame, from_=0.1, to=1.0, orient='horizontal', 
              variable=sens_var, style="Modern.Horizontal.TProgressbar").pack(fill='x')

    # Keywords Entry
    tk.Label(main_frame, text="Wake Keywords (comma-separated)", font=MODERN_FONT_BOLD, 
             bg=MODERN_BG, fg=MODERN_TEXT).pack(anchor='w', pady=(15, 5))
    kw_entry = tk.Entry(main_frame, textvariable=kw_var, bg=MODERN_SURFACE, 
                       fg=MODERN_TEXT, insertbackground=MODERN_ACCENT, 
                       relief='flat', font=MODERN_FONT)
    kw_entry.pack(fill='x', ipady=5)

    def save_settings():
        global entertainment_active
        conn = _ensure_secrets_db()
        master = _prompt_master(win)
        if not master:
            messagebox.showerror("Settings", "Master password required.")
            conn.close()
            return
        if not _verify_master(conn, master):
            messagebox.showerror("Settings", "Invalid master password.")
            conn.close()
            return
        conn.close()

        entertainment_active = ent_var.get()
        data = {
            "autostart": bool(auto_var.get()),
            "background": bool(bg_var.get()),
            "wake_sensitivity": float(sens_var.get()),
            "wake_keywords": kw_var.get().strip(),
            "offline_stt": bool(offline_var.get()),
            "require_strong_vault": bool(strong_vault_var.get())
        }
        _save_settings(data)
        if data["autostart"]: ensure_autostart()
        try: restart_wake_listener()
        except Exception: pass
        messagebox.showinfo("Settings", "Saved. Settings applied.")
        win.destroy()

    btn_frame = tk.Frame(main_frame, bg=MODERN_BG)
    btn_frame.pack(fill='x', side='bottom', pady=(20, 0))

    ttk.Button(btn_frame, text="Save Settings", style="ModernAccent.TButton", 
               command=save_settings).pack(fill='x', pady=(0, 10))
    ttk.Button(btn_frame, text="View Logs", style="Modern.TButton", 
               command=lambda: open_diagnostics_window(win)).pack(fill='x')

def open_macros_ui(parent=None):
    macros = _macros_load()
    win = tk.Toplevel(parent) if parent else tk.Tk()
    win.title("Sentinel Macros")
    win.geometry("520x640")
    win.configure(bg=MODERN_BG)

    main_frame = tk.Frame(win, bg=MODERN_BG, padx=20, pady=20)
    main_frame.pack(fill='both', expand=True)

    tk.Label(main_frame, text="MACRO MANAGER", font=MODERN_FONT_LARGE, 
             bg=MODERN_BG, fg=MODERN_ACCENT).pack(anchor='w', pady=(0, 20))

    # List of existing macros
    list_frame = tk.Frame(main_frame, bg=MODERN_BG)
    list_frame.pack(fill='both', expand=True, pady=(0, 15))

    tk.Label(list_frame, text="Saved Macros", font=MODERN_FONT_BOLD, 
             bg=MODERN_BG, fg=MODERN_TEXT).pack(anchor='w', pady=(0, 5))

    lst = tk.Listbox(list_frame, bg=MODERN_SURFACE, fg=MODERN_TEXT, 
                    selectbackground=MODERN_ACCENT, selectforeground=MODERN_BG,
                    relief='flat', borderwidth=0, font=MODERN_FONT,
                    highlightthickness=1, highlightbackground="#334155")
    lst.pack(fill='both', expand=True)
    for k in macros.keys():
        lst.insert('end', k)

    steps_txt = tk.Text(main_frame, height=4, bg=MODERN_SURFACE, fg=MODERN_TEXT,
                       insertbackground=MODERN_ACCENT, relief='flat', 
                       font=("Consolas", 10), padx=10, pady=10,
                       highlightthickness=1, highlightbackground="#334155")
    steps_txt.pack(fill='x', pady=(0, 15))

    def show_steps(evt=None):
        sel = lst.curselection()
        if not sel: return
        name = lst.get(sel[0])
        steps = macros.get(_normalize_name(name), [])
        steps_txt.delete('1.0', 'end')
        steps_txt.insert('1.0', "\n".join(steps))

    lst.bind('<<ListboxSelect>>', show_steps)

    def run_sel():
        sel = lst.curselection()
        if not sel: return
        name = lst.get(sel[0])
        steps = macros.get(_normalize_name(name), [])
        speak(f"Running macro {name}")
        def run_steps():
            for s in steps:
                try:
                    execute_command(s)
                    time.sleep(0.8)
                except Exception: pass
        threading.Thread(target=run_steps, daemon=True).start()

    def delete_sel():
        sel = lst.curselection()
        if not sel: return
        name = lst.get(sel[0])
        m = _macros_load()
        if m.pop(_normalize_name(name), None) is not None and _macros_save(m):
            speak("Macro deleted")
            lst.delete(sel[0])
            steps_txt.delete('1.0', 'end')

    btn_row = tk.Frame(main_frame, bg=MODERN_BG)
    btn_row.pack(fill='x', pady=(0, 20))
    ttk.Button(btn_row, text="Run Macro", style="ModernAccent.TButton", 
               command=run_sel).pack(side='left', expand=True, fill='x', padx=(0, 5))
    ttk.Button(btn_row, text="Delete", style="Modern.TButton", 
               command=delete_sel).pack(side='right', expand=True, fill='x', padx=(5, 0))

    # Add New Macro Section
    add_frame = tk.Frame(main_frame, bg=MODERN_SURFACE, padx=15, pady=15)
    add_frame.pack(fill='x')

    tk.Label(add_frame, text="Add New Macro", font=MODERN_FONT_BOLD, 
             bg=MODERN_SURFACE, fg=MODERN_TEXT).pack(anchor='w', pady=(0, 10))

    name_var = tk.StringVar()
    steps_var = tk.StringVar()
    
    tk.Label(add_frame, text="Name", font=("Segoe UI Variable Display", 8), 
             bg=MODERN_SURFACE, fg=MODERN_TEXT_MUTED).pack(anchor='w')
    tk.Entry(add_frame, textvariable=name_var, bg=MODERN_BG, fg=MODERN_TEXT, 
             relief='flat', insertbackground=MODERN_ACCENT).pack(fill='x', pady=(0, 10), ipady=3)

    tk.Label(add_frame, text="Steps (semicolon-separated)", font=("Segoe UI Variable Display", 8), 
             bg=MODERN_SURFACE, fg=MODERN_TEXT_MUTED).pack(anchor='w')
    tk.Entry(add_frame, textvariable=steps_var, bg=MODERN_BG, fg=MODERN_TEXT, 
             relief='flat', insertbackground=MODERN_ACCENT).pack(fill='x', pady=(0, 15), ipady=3)

    def save_new():
        name = name_var.get().strip()
        steps = [s.strip() for s in steps_var.get().split(';') if s.strip()]
        if not name or not steps: return
        m = _macros_load()
        m[_normalize_name(name)] = steps
        if _macros_save(m):
            speak("Macro saved")
            lst.insert('end', name)
            name_var.set(""); steps_var.set("")

    ttk.Button(add_frame, text="Save New Macro", style="ModernAccent.TButton", 
               command=save_new).pack(fill='x')

def open_diagnostics_window(parent=None):
    try:
        lp = os.path.join(APPDATA_DIR, "sentinel.log")
        log_text = ""
        if os.path.exists(lp):
            with open(lp, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[-400:]
                log_text = "".join(lines)
        
        lv = tk.Toplevel(parent) if parent else tk.Tk()
        lv.title("Sentinel Diagnostics")
        lv.geometry("680x520")
        lv.configure(bg=MODERN_BG)

        main_frame = tk.Frame(lv, bg=MODERN_BG, padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)

        tk.Label(main_frame, text="DIAGNOSTICS & LOGS", font=MODERN_FONT_LARGE, 
                 bg=MODERN_BG, fg=MODERN_ACCENT).pack(anchor='w', pady=(0, 20))

        txt_frame = tk.Frame(main_frame, bg=MODERN_BG)
        txt_frame.pack(fill='both', expand=True, pady=(0, 20))

        txt = tk.Text(txt_frame, wrap='none', bg=MODERN_SURFACE, fg=MODERN_TEXT,
                     relief='flat', font=("Consolas", 9), padx=10, pady=10,
                     highlightthickness=1, highlightbackground="#334155")
        txt.insert('1.0', log_text or "No logs found.")
        txt.configure(state='disabled')
        txt.pack(fill='both', expand=True)

        # Mic Level in Diagnostics
        mic_frame = tk.Frame(main_frame, bg=MODERN_SURFACE, padx=15, pady=15)
        mic_frame.pack(fill='x')

        tk.Label(mic_frame, text="Real-time Mic Monitor", font=MODERN_FONT_BOLD, 
                 bg=MODERN_SURFACE, fg=MODERN_TEXT).pack(anchor='w', pady=(0, 10))

        level_var = tk.DoubleVar(value=0.0)
        bar = ttk.Progressbar(mic_frame, orient='horizontal', mode='determinate', 
                             maximum=1.0, variable=level_var, style="Modern.Horizontal.TProgressbar")
        bar.pack(fill='x', height=10)

        def update_level():
            try:
                pa = pyaudio.PyAudio()
                stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                                 input=True, frames_per_buffer=512)
                while True:
                    if not lv.winfo_exists(): break
                    data = stream.read(512, exception_on_overflow=False)
                    samples = np.frombuffer(data, dtype=np.int16)
                    amp = float(np.mean(np.abs(samples))) / 32768.0
                    level_var.set(min(1.0, amp * 12.0))
                    time.sleep(0.05)
            except Exception: pass
            finally:
                try: stream.close(); pa.terminate()
                except Exception: pass

        threading.Thread(target=update_level, daemon=True).start()
        
        ttk.Button(main_frame, text="Close", style="Modern.TButton", 
                   command=lv.destroy).pack(fill='x', pady=(20, 0))

    except Exception as e:
        print(f"Error opening diagnostics: {e}")
        # health statuses
        health = []
        health.append("OpenAI: OK" if os.getenv("OPEN_AI_API_KEY") else "OpenAI: Missing")
        health.append("Porcupine: OK" if ACCESS_KEY else "Porcupine: Missing")
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=2)
            health.append("Network: OK")
        except Exception:
            health.append("Network: Unavailable")
        ttk.Label(frm, text=" | ".join(health)).pack(pady=6)
    except Exception:
        if messagebox:
            messagebox.showerror("Diagnostics", "Unable to open logs.")
NOTES_PATH = os.path.join(APPDATA_DIR, "notes.json")

def _notes_add(text):
    ensure_appdata_dir()
    notes = []
    try:
        if os.path.exists(NOTES_PATH):
            with open(NOTES_PATH, "r", encoding="utf-8") as f:
                notes = json.load(f)
    except Exception:
        notes = []
    notes.append(text)
    try:
        with open(NOTES_PATH, "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=2)
        return True
    except Exception:
        return False

def _notes_list():
    try:
        if os.path.exists(NOTES_PATH):
            with open(NOTES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return []
    return []

def clean_downloads_folder():
    try:
        path = os.path.join(os.path.expanduser("~"), "Downloads")
        count = 0
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                # Only delete old files (> 30 days) to be safe
                if time.time() - os.path.getmtime(fp) > 86400 * 30:
                    os.remove(fp)
                    count += 1
        status_queue.put(f"Cleaned {count} files from Downloads.")
        speak(f"Cleaned {count} files from your downloads folder.")
    except Exception as e:
        logging.error(f"Error cleaning downloads: {e}")

def find_large_files_in_downloads():
    try:
        path = os.path.join(os.path.expanduser("~"), "Downloads")
        files = []
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                size = os.path.getsize(fp)
                if size > 100 * 1024 * 1024: # > 100MB
                    files.append((f, size / (1024 * 1024)))
        
        files.sort(key=lambda x: x[1], reverse=True)
        if not files:
            speak("No files larger than 100 megabytes found.")
        else:
            speak(f"Found {len(files)} large files. The largest is {files[0][0]} at {int(files[0][1])} megabytes.")
            for f, s in files[:3]:
                print(f"[Large File] {f} ({int(s)}MB)")
    except Exception as e:
        logging.error(f"Error finding large files: {e}")
def ensure_vosk_model():
    model_dir = os.path.join(APPDATA_DIR, "vosk-model")
    if os.path.isdir(model_dir) and os.listdir(model_dir):
        return True
    try:
        import urllib.request, zipfile, io
        url = os.getenv("VOSK_DL") or "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        data = urllib.request.urlopen(url, timeout=30).read()
        z = zipfile.ZipFile(io.BytesIO(data))
        target = model_dir
        os.makedirs(target, exist_ok=True)
        for m in z.namelist():
            if m.endswith('/'):
                continue
            rel = m.split('/', 1)[1] if '/' in m else m
            dest = os.path.join(target, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(m) as src, open(dest, 'wb') as out:
                out.write(src.read())
        return True
    except Exception:
        return False

if __name__ == "__main__":
    import sys
    if "--install-deps" in sys.argv:
        try:
            print("Installing Playwright browsers...")
            subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
            sys.exit(0)
        except Exception as e:
            print(f"Error installing dependencies: {e}")
            sys.exit(1)
    main()
