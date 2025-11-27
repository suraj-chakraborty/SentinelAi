#
# sentinel_ai_fixed.py
import os
import time
import wave
import threading
import webbrowser
import subprocess
from queue import Queue
from dotenv import load_dotenv

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
from pathlib import Path
import logging

# Speaker embedding utilities
# Silero VAD utils
 

# GUI
import tkinter as tk
from tkinter import messagebox, ttk
try:
    import pystray
    from PIL import Image, ImageDraw
    _tray_available = True
except Exception:
    _tray_available = False

load_dotenv()

# ---------- Configuration ----------
ACCESS_KEY = os.getenv("PVPORCUPINE_PRIVATE_KEY")  # picovoice key
PORCUPINE_KEYWORD_PATH = "./assets/sounds/Hey-robert_en_windows_v3_0_0.ppn"
REFERENCE_WAV = "reference.wav"
OWNER_EMBED_PATH = "owner_embed.npy"
COMMAND_WAV = "command.wav"
LIVENESS_WAV = "liveness.wav"

# status queue for GUI updates
status_queue = Queue()

agent_active = False
pending_step = None
next_agent_capture_time = 0
entertainment_active = False

# App registry paths
APPDATA_DIR = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi")
APPS_REGISTRY_PATH = os.path.join(APPDATA_DIR, "apps_registry.json")
apps_index = {}
_mru = {}
MRU_PATH = os.path.join(APPDATA_DIR, "mru.json")
SETTINGS_PATH = os.path.join(APPDATA_DIR, "settings.json")
wake_stop_event = threading.Event()
wake_thread = None
root_window = None
listening_blocked_until = 0
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
    salt, mh = _get_master_record(conn)
    if not salt or not mh:
        _set_master_record(conn, master)
        return True
    return hmac.compare_digest(_pbkdf(master, salt), mh)

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
                        if icon and str(icon).lower().endswith(".exe"):
                            exe = icon
                        elif loc and os.path.isdir(loc):
                            # try common exe names
                            candidates = [p for p in Path(loc).glob("*.exe")]
                            if candidates:
                                exe = str(candidates[0])
                        if name:
                            results[_normalize_name(name)] = exe or ""
                except OSError:
                    continue
    except OSError:
        pass
    return results

def build_app_registry():
    ensure_appdata_dir()
    idx = {}
    # HKLM and HKCU
    idx.update(_scan_uninstall_key(winreg.HKEY_LOCAL_MACHINE))
    idx.update(_scan_uninstall_key(winreg.HKEY_CURRENT_USER))
    # Known apps fallback
    known = {
        "chrome": r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "google chrome": r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "notepad": r"C:\\Windows\\System32\\notepad.exe",
        "paint": r"C:\\Windows\\System32\\mspaint.exe",
        "calculator": r"C:\\Windows\\System32\\calc.exe",
        "vscode": os.path.expandvars(r"%LocalAppData%\\Programs\\Microsoft VS Code\\Code.exe"),
        "visual studio code": os.path.expandvars(r"%LocalAppData%\\Programs\\Microsoft VS Code\\Code.exe"),
        "edge": r"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
        "brave": os.path.expandvars(r"%LocalAppData%\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"),
        "opera": os.path.expandvars(r"%LocalAppData%\\Programs\\Opera\\launcher.exe"),
        "vlc": r"C:\\Program Files\\VideoLAN\\VLC\\vlc.exe",
        "notepad++": r"C:\\Program Files\\Notepad++\\notepad++.exe",
        "spotify": os.path.expandvars(r"%LocalAppData%\\Microsoft\\WindowsApps\\Spotify.exe"),
        "steam": r"C:\\Program Files (x86)\\Steam\\steam.exe",
        "word": r"C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE",
        "excel": r"C:\\Program Files\\Microsoft Office\\root\\Office16\\EXCEL.EXE",
        "powerpoint": r"C:\\Program Files\\Microsoft Office\\root\\Office16\\POWERPNT.EXE",
    }
    for k, v in known.items():
        if k not in idx and os.path.exists(v):
            idx[k] = v
    # PATH scan for common executables
    try:
        for d in os.getenv("PATH", "").split(os.pathsep):
            d = d.strip('"')
            if not d:
                continue
            for name, exe in [("python", "python.exe"), ("git", "git.exe"), ("node", "node.exe"), ("vlc", "vlc.exe"), ("code", "Code.exe")]:
                p = os.path.join(d, exe)
                if os.path.exists(p):
                    idx[_normalize_name(name if name != "code" else "vscode")] = p
    except Exception:
        pass
    # Adobe scan
    try:
        ad = r"C:\\Program Files\\Adobe"
        if os.path.isdir(ad):
            for p in Path(ad).glob("**/Photoshop.exe"):
                idx["photoshop"] = str(p)
            for p in Path(ad).glob("**/Illustrator.exe"):
                idx["illustrator"] = str(p)
            for p in Path(ad).glob("**/Adobe Premiere*.exe"):
                idx["premiere"] = str(p)
            for p in Path(ad).glob("**/AfterFX.exe"):
                idx["aftereffects"] = str(p)
    except Exception:
        pass
    # Messaging scan
    try:
        wa = os.path.expandvars(r"%LocalAppData%\\WhatsApp\\WhatsApp.exe")
        if os.path.exists(wa):
            idx["whatsapp"] = wa
        tg = os.path.expandvars(r"%LocalAppData%\\Telegram Desktop\\Telegram.exe")
        if os.path.exists(tg):
            idx["telegram"] = tg
        for p in Path(os.path.expandvars(r"%LocalAppData%\\Discord")).glob("**/Discord.exe"):
            idx["discord"] = str(p)
    except Exception:
        pass
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
    exe = apps_index.get(n)
    if exe and os.path.exists(exe):
        return exe
    try:
        for k, v in apps_index.items():
            if n in k and v and os.path.exists(v):
                return v
    except Exception:
        pass
    return None

def open_app(name):
    exe = find_app_executable(name)
    if exe:
        try:
            subprocess.Popen([exe])
            status_queue.put(f"Opening {name}")
            speak(f"Opening {name}")
            record_mru(name)
            return True
        except Exception:
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
        build_app_registry()
    missing = []
    if not find_app_executable("google chrome"):
        missing.append("Google Chrome")
    if not find_app_executable("visual studio code") and not find_app_executable("vscode"):
        missing.append("Visual Studio Code")
    if missing:
        try:
            root = tk.Tk(); root.withdraw()
            ans = messagebox.askyesno("Install Tools", f"Install {', '.join(missing)}?")
            root.destroy()
            if ans:
                for m in missing:
                    install_and_open_app(m)
        except Exception:
            pass

def _make_tray_image():
    img = Image.new('RGB', (64, 64), color=(30, 30, 30))
    d = ImageDraw.Draw(img)
    d.ellipse((8,8,56,56), outline=(0,200,255), width=3)
    d.text((18,24), 'SA', fill=(255,255,255))
    return img

def start_tray():
    if not _tray_available:
        return
    def tray_start(icon, item):
        start_agent()
    def tray_stop(icon, item):
        stop_agent()
    def tray_toggle_ent(icon, item):
        global entertainment_active
        entertainment_active = not entertainment_active
        speak("Entertainment " + ("enabled" if entertainment_active else "disabled"))
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
                return
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
    def tray_settings(icon, item):
        try:
            open_settings_ui_global()
        except Exception:
            pass
    def tray_quit(icon, item):
        os._exit(0)
    menu = pystray.Menu(
        pystray.MenuItem('Start Agent', tray_start),
        pystray.MenuItem('Stop Agent', tray_stop),
        pystray.MenuItem('Record Command', tray_record),
        pystray.MenuItem('Toggle Entertainment', tray_toggle_ent),
        pystray.MenuItem('Settings', tray_settings),
        pystray.MenuItem('Quit', tray_quit)
    )
    icon = pystray.Icon('SentinelAI', _make_tray_image(), 'SentinelAI', menu)
    threading.Thread(target=icon.run, daemon=True).start()

def start_agent():
    # Turn on the step-by-step helper. It will wait for steps and ask for confirmation.
    global agent_active, pending_step
    agent_active = True
    pending_step = None
    status_queue.put("Agent active. Describe the first step.")
    speak("Agent started. Describe the first step.")

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
    speak("Provide a step or say confirm.")

def gemini_generate(prompt):
    # Simple helper that asks Gemini (Google AI) for a response.
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return "Gemini key missing."
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + key
    body = {"contents": [{"parts": [{"text": str(prompt)}]}]}
    try:
        with httpx.Client(timeout=30) as client:
            r = client.post(url, json=body)
            j = r.json()
        return j.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "") or ""
    except Exception:
        return "Gemini error."

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
    x, sr = sf.read(filepath)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if len(x) < sr // 2:
        pad = sr // 2 - len(x)
        x = np.pad(x, (0, pad))
    x[1:] = x[1:] - 0.97 * x[:-1]
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    n_fft = 512 if sr <= 22050 else 1024
    fb = _mel_filterbank(n_fft, sr, n_mels=40)
    frames = []
    for start in range(0, len(x) - frame_len + 1, hop):
        frame = x[start:start + frame_len]
        frame = frame * np.hamming(frame_len)
        spec = np.fft.rfft(frame, n=n_fft)
        ps = (np.abs(spec) ** 2)
        mel = fb.dot(ps[:fb.shape[1]])
        mel = np.log(mel + 1e-10)
        frames.append(mel)
    if not frames:
        return np.zeros(fb.shape[0] * 2, dtype=np.float32)
    M = np.vstack(frames)
    mu = M.mean(axis=0)
    sigma = M.std(axis=0)
    emb = np.concatenate([mu, sigma]).astype(np.float32)
    return emb

# Silero VAD model (loads on first use)
 

# ---------- Utilities ----------

# Thread-safe TTS (non-blocking)
_tts_lock = threading.Lock()
_tts_disabled = False
def speak(text):
    global _tts_disabled
    def _beep():
        try:
            winsound.Beep(600, 200)
        except Exception:
            pass
    # Check disk space before attempting TTS
    try:
        free_bytes = shutil.disk_usage(os.path.dirname(__file__)).free
        if free_bytes < 10 * 1024 * 1024:
            _tts_disabled = True
    except Exception:
        pass
    if _tts_disabled:
        threading.Thread(target=_beep, daemon=True).start()
        return
    def _s():
        try:
            with _tts_lock:
                pythoncom.CoInitialize()
                engine = pyttsx3.init()
                engine.say(str(text))
                engine.runAndWait()
                pythoncom.CoUninitialize()
        except OSError:
            _tts_disabled = True
            _beep()
        except Exception as e:
            _beep()
    threading.Thread(target=_s, daemon=True).start()

def show_recording_overlay(parent=None):
    # Create a lightweight overlay window with a waveform canvas
    # to visualize microphone amplitude during recording.
    global recording_overlay, recording_canvas, recording_wave_values
    try:
        p = parent or root_window
        win = tk.Toplevel(p) if p else tk.Tk()
        win.title("Listening...")
        win.geometry("320x90")
        ttk.Label(win, text="Listening...").pack(pady=6)
        canvas = tk.Canvas(win, width=500, height=20, bg="#101010", highlightthickness=0)
        canvas.pack(padx=10, pady=6)
        recording_overlay = win
        recording_canvas = canvas
        recording_wave_values = []
    except Exception:
        recording_overlay = None
        recording_canvas = None
        recording_wave_values = []

def update_recording_overlay(amp):
    # Draw a scrolling waveform based on recent amplitude values.
    try:
        if recording_canvas is None:
            return
        w = int(recording_canvas.winfo_width())
        h = int(recording_canvas.winfo_height())
        # Normalize amplitude into [0,1] range and keep a short history
        a = max(0.0, min(1.0, float(amp) * 6.0))
        recording_wave_values.append(a)
        max_points = max(60, w)  # keep roughly one point per pixel
        if len(recording_wave_values) > max_points:
            recording_wave_values[:] = recording_wave_values[-max_points:]
        # Clear and redraw polyline
        recording_canvas.delete("wave")
        if len(recording_wave_values) > 1:
            step = max(1, int(max_points / len(recording_wave_values)))
            points = []
            for i, v in enumerate(recording_wave_values):
                x = int(i * (w / float(len(recording_wave_values))))
                y = int(h - (v * h))
                points.append((x, y))
            # Draw lines connecting points
            for i in range(1, len(points)):
                x1, y1 = points[i-1]
                x2, y2 = points[i]
                recording_canvas.create_line(x1, y1, x2, y2, fill="#12d46b", width=2, tags="wave")
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
def record_until_silence(filename, max_duration=25, samplerate=16000, channels=1, frames_per_buffer=1024, min_duration=4.0, silence_threshold=0.008, speech_threshold=0.015, silence_duration=0.8, on_amp=None):
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
                # continue collecting baseline without affecting quiet_t
                continue
            if baseline_vals:
                base = np.median(baseline_vals)
                base = max(base, 0.002)
                sp_thr = max(speech_threshold, base * 2.5)
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
            if elapsed >= min_duration:
                if not had_speech:
                    # If we never saw speech beyond threshold but peak is notably above baseline, accept.
                    if peak_amp >= sp_thr * 0.9:
                        had_speech = True
                    else:
                        break
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

# Liveness registration / check (Resemblyzer embeddings)
def register_reference_if_missing():
    # If we don't have the owner's voice saved yet,
    # we record a short sample and create its fingerprint.
    if not os.path.exists(OWNER_EMBED_PATH):
        speak("No registered voice found. Please say the passphrase after the beep.")
        time.sleep(0.6)
        print("[Recording reference voice]")
        record_wav(REFERENCE_WAV, duration=3)
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
        speak("Please say your reference passphrase after the beep.")
        time.sleep(0.6)

        status_queue.put("Recording new reference voice...")
        print("[Manual registration] Recording new reference...")

        record_wav(REFERENCE_WAV, duration=3)

        owner_embed = compute_embedding(REFERENCE_WAV)
        np.save(OWNER_EMBED_PATH, owner_embed)

        speak("Voice registration updated successfully.")
        status_queue.put("Voice registration updated successfully.")

        print("[Manual registration] Updated owner_embed.npy")
    except Exception as e:
        speak("Voice registration failed.")
        status_queue.put(f"Registration error: {e}")
        print("[Manual registration error]", e)

def liveness_check(threshold=0.80):
    # Checks quickly if the new voice sample looks like the owner's
    # by comparing their fingerprints.
    # Ensure reference exists
    if not os.path.exists(OWNER_EMBED_PATH):
        register_reference_if_missing()

    speak("Please repeat the passphrase after the beep.")
    time.sleep(0.5)
    print("[Recording liveness sample]")
    record_wav(LIVENESS_WAV, duration=3)
    try:
        live_embed = compute_embedding(LIVENESS_WAV)
        owner_embed = np.load(OWNER_EMBED_PATH)
        similarity = _cosine(live_embed, owner_embed)
        print(f"[Liveness Similarity Score]: {similarity:.4f}")
        status_queue.put(f"Liveness score: {similarity:.3f}")
        return similarity >= threshold
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
    # Turns a voice recording into text using Google's free speech tool.
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        print("[Transcribed]:", text)
        return text.lower()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        off = transcribe_wav_offline(filename)
        if off:
            print("[Offline Transcribed]:", off)
        return off

# Interpret command using OpenAI (simple wrapper)
def interpret_command(command):
    # If you say: "ask gemini something", we ask Gemini (Google AI).
    # Otherwise we ask OpenAI to summarize the command.
    c = (command or "").lower()
    if c.startswith("ask gemini") or c.startswith("use gemini"):
        q = command.split(" ", 2)
        prompt = q[2] if len(q) > 2 else command
        return gemini_generate(prompt)
    if entertainment_active:
        # In entertainment mode, respond conversationally using OpenAI
        from openai import OpenAI
        key = os.getenv("OPEN_AI_API_KEY")
        if not key:
            return "No OpenAI key configured."
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":"You are a friendly, concise, kid-safe desktop companion named SentinelAI. Keep replies short unless asked to expand."},
                    {"role":"user","content":command}
                ],
                max_tokens=120
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return "Chat error."
    from openai import OpenAI
    key = os.getenv("OPEN_AI_API_KEY")
    if not key:
        return "No OpenAI key configured."
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,)
    prompt = f"You are a helpful desktop assistant. Convert the command into a short action summary. Command: {command}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a helpful desktop assistant."},
                {"role":"user","content":prompt}
            ],
            max_tokens=120
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        import openai
        openai.api_key = key
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role":"system","content":"You are a helpful desktop assistant."},
                {"role":"user","content":prompt}
            ],
            max_tokens=120
        )
        return resp.choices[0].message.content.strip()

# Execute a handful of commands (keeps your original behaviors)
def execute_command(command):
    # This is the action center. It reads the command text and does the matching thing.
    original = command or ""
    command = (command or "").lower()
    print("[Execute] ", command)
    global agent_active
    if agent_active:
        agent_handle(command)  # When agent is active, treat every sentence as a step.
        return
    if command in ("start", "start agent", "agent start", "enable agent"):
        speak("Starting agent")
        start_agent()  # Turn on agent mode
        return
    if command in ("stop", "stop agent", "agent stop", "disable agent"):
        speak("Stopping agent")
        stop_agent()   # Turn off agent mode
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
    # Secrets: store
    if ("password for" in command or "variable" in command) and ("keep in mind" in command or "remember" in command or "store" in command) and (" is " in command or " = " in command):
        base = "password for"
        idx = command.find(base)
        if idx == -1:
            base = "variable"
            idx = command.find(base)
        end = command.find(" is ", idx)
        sep = 4
        if end == -1:
            end = command.find(" = ", idx)
            sep = 3
        name = original[idx+len(base):end].strip()
        pwd = original[end+sep:].strip()
        if name and pwd:
            ok = store_secret(_normalize_name(name), pwd, parent=root_window)
            if ok:
                speak(f"Stored password for {name}")
        return
    # Secrets: fetch
    if ("password for" in command or "variable" in command) and ("give me" in command or "what is" in command or "show" in command):
        base = "password for"
        idx = command.find(base)
        if idx == -1:
            base = "variable"
            idx = command.find(base)
        name = original[idx+len(base):].strip()
        if name:
            pw = fetch_secret(_normalize_name(name), parent=root_window)
            if pw:
                speak_password_spelled(pw)
            else:
                speak("No password found")
        return

    if command.startswith("open "):
        appname = command.split("open ", 1)[1].strip()
        if appname:
            speak(f"Opening {appname}")
        if not open_app(appname):
            install_and_open_app(appname)
    elif "open browser" in command:
        speak("Opening browser")
        webbrowser.open("https://www.google.com")
    elif "open gmail" in command:
        speak("Opening Gmail")
        webbrowser.open("https://mail.google.com")

    # Notes management
    elif command.startswith("remember note ") or command.startswith("save note "):
        note = original.split(" ", 2)[2].strip()
        if note:
            _notes_add(note)
            speak("Note saved")
        return
    elif command in ("list notes", "show notes"):
        items = _notes_list()
        if not items:
            speak("No notes")
        else:
            speak("You have " + str(len(items)) + " notes")
            for i, n in enumerate(items, 1):
                speak(f"Note {i}: {n[:80]}")
        return
    elif command.startswith("forget note "):
        idx_s = command.split("forget note ", 1)[1].strip()
        try:
            idx = int(idx_s)
            if _notes_forget(idx):
                speak("Note removed")
            else:
                speak("No such note")
        except Exception:
            speak("Please provide a valid note number")
        return

    elif command in ("list commands", "help commands", "what can you do"):
        speak("You can say open app names, start or stop agent, query system info, battery report, and manage secrets by saying remember password for name is value, or give me password for name")
 

import os
import subprocess
import webbrowser
import pyttsx3
import speech_recognition as sr
import numpy as np
 
 
import soundfile as sf
import time
import tkinter as tk
from tkinter import messagebox, simpledialog
import threading
import pvporcupine
import struct
import pyaudio
import openai
from queue import Queue
from dotenv import load_dotenv, dotenv_values
import sqlite3
import hashlib
import hmac
import ctypes
from ctypes import wintypes

load_dotenv() 


status_queue = Queue()

# Set your OpenAI API key
openai.api_key = os.getenv("OPEN_AI_API_KEY")  # Or replace with your key string

 

# Initialize TTS engine
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Record audio to file
def record_audio_to_file(filename="liveness.wav", duration=3):
    samplerate = 16000
    print(f"[Recording] Saving to {filename} for {duration} seconds...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, recording, samplerate)
    print("[Recording Finished]")
    print(f"[Saved] {filename}")
    return filename


def liveness_check(threshold=0.80):
    # Uses simple audio embedding similarity to verify live voice vs owner sample
    if not os.path.exists(OWNER_EMBED_PATH):
        register_reference_if_missing()
    speak("Please repeat the phrase: I am the master.")
    record_wav(LIVENESS_WAV, duration=3)
    try:
        trim_wav_silence(LIVENESS_WAV)
    except Exception:
        pass
    try:
        live_embed = compute_embedding(LIVENESS_WAV)
        owner_embed = np.load(OWNER_EMBED_PATH)
        similarity = _cosine(live_embed, owner_embed)
        print(f"[Liveness Similarity Score]: {similarity:.4f}")
        return similarity >= threshold
    except Exception as e:
        print(f"[Error in liveness check]: {e}")
        return False

# Transcribe command
def get_command():
    recognizer = sr.Recognizer()
    with sr.AudioFile("liveness.wav") as source:
        audio = recognizer.record(source)
    try:
        command = recognizer.recognize_google(audio)
        print("Google Speech Recognition thinks you said " + recognizer.recognize_google(audio))
        print("command:", command)
        return command.lower()
    except sr.UnknownValueError:
        print("Google Speech Recognition thinks you said " + recognizer.recognize_google(audio))
        speak("Sorry, I did not understand.")
        return ""
    except sr.RequestError:
        print("Google Speech Recognition thinks you said " + recognizer.recognize_google(audio))
        speak("Sorry, my speech service is currently unavailable.")
        return ""

# Use OpenAI to interpret command and generate a response or action

def interpret_command(command):
    print("command:", command)
    prompt = f"You are a smart desktop assistant. Interpret the user's voice command and suggest a Python function call.\nCommand: {command}\nRespond with the best matching action."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for executing desktop tasks."},
                {"role": "user", "content": prompt}
            ]
        )
        action = response.choices[0].message.content.strip()
        return action
    except Exception as e:
        speak("I encountered an error accessing my intelligence service.")
        return ""

# Execute interpreted action
def execute_command(command):
    original = command or ""
    lc = original.lower()
    if "open browser" in command:
        webbrowser.open("https://www.google.com")
    elif "open gmail" in command:
        webbrowser.open("https://www.google.com/gmail")
 
    elif "open youtube" in command:
        webbrowser.open("https://www.youtube.com")
    elif "open facebook" in command:
        webbrowser.open("https://www.facebook.com")
    elif "open github" in command:
        webbrowser.open("https://www.github.com")
 
    elif "open settings" in command:
        os.system("start ms-settings:")
    elif "open task manager" in command or "task manager" in command:
        subprocess.Popen(["taskmgr"])
    elif "open powershell" in command:
        subprocess.Popen(["powershell.exe"])
    elif "open vscode" in command or "open vs code" in command:
        paths = [
            os.path.expandvars(r"%LocalAppData%\Programs\Microsoft VS Code\Code.exe"),
            r"C:\\Program Files\\Microsoft VS Code\\Code.exe"
        ]
        exe = next((p for p in paths if os.path.exists(p)), None)
        if exe:
            subprocess.Popen([exe])
        else:
            try:
                subprocess.Popen(["code"])
            except Exception:
                speak("VS Code not found.")
    elif "system info" in command:
        try:
            out = subprocess.check_output(["systeminfo"], shell=True, text=True, timeout=20)
            status_queue.put("System info collected.")
            print(out[:1000])
            speak("System information collected.")
        except Exception:
            status_queue.put("System info failed.")
    elif "ip address" in command or "show ip" in command:
        try:
            out = subprocess.check_output(["ipconfig"], shell=True, text=True, timeout=15)
            lines = [l.strip() for l in out.splitlines() if "IPv4" in l]
            msg = "; ".join(lines) or "No IPv4 found."
            status_queue.put(msg)
            speak("IP information updated.")
        except Exception:
            status_queue.put("IP check failed.")
    elif "battery report" in command:
        try:
            report = os.path.join(os.path.dirname(__file__), "battery-report.html")
            subprocess.check_call(["powercfg", "/batteryreport", "/output", report], shell=True)
            os.startfile(report)
            status_queue.put("Battery report opened.")
            speak("Battery report opened.")
        except Exception:
            status_queue.put("Battery report failed.")
    elif command.startswith("ask gemini") or command.startswith("use gemini"):
        q = command.split(" ", 2)
        prompt = q[2] if len(q) > 2 else command
        resp = gemini_generate(prompt)
        speak(resp)
    elif "start agent" in command or command.startswith("agent"):
        start_agent()
    elif "stop agent" in command:
        stop_agent()
    elif "open gpt" in command:
        try:
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            import speech_recognition as sr
            import time
        except Exception:
            speak("Web automation unavailable.")
            return
 
    elif "open gpt" in command:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        import speech_recognition as sr
        import time
 

        # Set up Chrome options
        options = Options()
        options.add_argument("--start-maximized")  # Open browser maximized

        # Start driver
        service = Service("chromedriver")  # Adjust if not in PATH
        driver = webdriver.Chrome(service=service, options=options)
        driver.get("https://chatgpt.com/?model=auto")

        # Wait for page load
        wait = WebDriverWait(driver, 30)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "textarea")))

        # Type voice input
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Speak your message to ChatGPT:")
            audio = recognizer.listen(source)

        try:
            query = recognizer.recognize_google(audio)
            print(f"You said: {query}")
            textarea = driver.find_element(By.TAG_NAME, "textarea")
            textarea.send_keys(query)
            time.sleep(1)

            # Click the submit button using its ID
            send_button = driver.find_element(By.ID, "composer-submit-button")
            send_button.click()
            print("Query sent to ChatGPT.")

        except sr.UnknownValueError:
            print("Sorry, could not understand your voice.")
        except Exception as e:
            print("Error:", e)


    elif "shutdown" in command:
        speak("Shutting down. Goodbye.")
        os.system("shutdown /s /t 1")
    elif "restart" in command:
        os.system("shutdown /r /t 1")
    elif "open notepad" in command or "notepad" in command:
        subprocess.Popen(["notepad.exe"])
    elif "open chrome" in command or "chrome" in command:
        query = command.replace("open chrome", "").replace("search", "").strip() or "python programming"
        webbrowser.open(f"https://www.google.com/search?q={query.replace(' ', '+')}" )
        speak(f"Searching Chrome for {query}")
    # Window management
    elif command in ("minimize window", "window minimize"):
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            ctypes.windll.user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
            speak("Window minimized")
        except Exception:
            speak("Unable to minimize window")
    elif command in ("maximize window", "window maximize"):
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
            speak("Window maximized")
        except Exception:
            speak("Unable to maximize window")
    elif command in ("close window", "window close"):
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
            speak("Window closed")
        except Exception:
            speak("Unable to close window")
    elif command in ("snap left", "window left"):
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            sw = user32.GetSystemMetrics(0)
            sh = user32.GetSystemMetrics(1)
            user32.MoveWindow(hwnd, 0, 0, int(sw/2), sh, True)
            speak("Snapped left")
        except Exception:
            speak("Unable to snap left")
    elif command in ("snap right", "window right"):
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            sw = user32.GetSystemMetrics(0)
            sh = user32.GetSystemMetrics(1)
            user32.MoveWindow(hwnd, int(sw/2), 0, int(sw/2), sh, True)
            speak("Snapped right")
        except Exception:
            speak("Unable to snap right")
    elif command in ("move window to left monitor", "window left monitor"):
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            vx = user32.GetSystemMetrics(76)
            vy = user32.GetSystemMetrics(77)
            vw = user32.GetSystemMetrics(78)
            vh = user32.GetSystemMetrics(79)
            user32.MoveWindow(hwnd, vx, vy, int(vw/2), vh, True)
            speak("Moved to left monitor")
        except Exception:
            speak("Unable to move window")
    elif command in ("move window to right monitor", "window right monitor"):
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            vx = user32.GetSystemMetrics(76)
            vy = user32.GetSystemMetrics(77)
            vw = user32.GetSystemMetrics(78)
            vh = user32.GetSystemMetrics(79)
            user32.MoveWindow(hwnd, vx + int(vw/2), vy, int(vw/2), vh, True)
            speak("Moved to right monitor")
        except Exception:
            speak("Unable to move window")
    elif "clean downloads" in command:
        d = os.path.join(os.path.expanduser("~"), "Downloads")
        count = 0
        try:
            for rootp, dirs, files in os.walk(d):
                for f in files:
                    fp = os.path.join(rootp, f)
                    try:
                        os.remove(fp)
                        count += 1
                    except Exception:
                        pass
            speak(f"Cleaned {count} files from Downloads")
        except Exception:
            speak("Unable to clean Downloads")
    elif "find large files" in command:
        d = os.path.join(os.path.expanduser("~"), "Downloads")
        results = []
        try:
            for rootp, dirs, files in os.walk(d):
                for f in files:
                    fp = os.path.join(rootp, f)
                    try:
                        sz = os.path.getsize(fp)
                        if sz > 100 * 1024 * 1024:
                            results.append((fp, sz))
                    except Exception:
                        pass
            results.sort(key=lambda x: x[1], reverse=True)
            if not results:
                speak("No large files found in Downloads")
            else:
                speak("Top large files")
                for fp, sz in results[:5]:
                    speak(os.path.basename(fp) + " size " + str(int(sz/1024/1024)) + " megabytes")
        except Exception:
            speak("Unable to scan Downloads")
    else:
        if lc in ("stop listening", "pause listening", "do not listen"):
            global listening_blocked_until
            listening_blocked_until = time.time() + 300
            speak("Pausing listening for five minutes")
            return
        if lc in ("start listening", "resume listening"):
            listening_blocked_until = 0
            speak("Listening resumed")
            return
        if lc in ("list commands", "help commands", "what can you do"):
            speak("You can say open apps, start or stop agent, system info, battery report, manage secrets by saying remember password for name is value or give me password for name")
            return
        # Secrets handling (store/fetch)
        if ("password for" in lc or "variable" in lc) and ("keep in mind" in lc or "remember" in lc or "store" in lc) and (" is " in lc or " = " in lc):
            base = "password for" if "password for" in lc else "variable"
            idx = lc.find(base)
            end = lc.find(" is ", idx)
            sep = 4
            if end == -1:
                end = lc.find(" = ", idx)
                sep = 3
            name = original[idx+len(base):end].strip()
            pwd = original[end+sep:].strip()
            if name and pwd:
                ok = store_secret(_normalize_name(name), pwd, parent=root_window)
                if ok:
                    speak(f"Stored password for {name}")
            return
        if ("password for" in lc or "variable" in lc) and ("give me" in lc or "what is" in lc or "show" in lc):
            base = "password for" if "password for" in lc else "variable"
            idx = lc.find(base)
            name = original[idx+len(base):].strip()
            if name:
                pw = fetch_secret(_normalize_name(name), parent=root_window)
                if pw:
                    speak_password_spelled(pw)
                else:
                    speak("No password found")
            return
        # fallback interpreter + speak
        resp = interpret_command(command)
        speak(resp)

# ---------- Wake-word listener (runs in background thread) ----------
def listen_for_wake_word_loop(session_duration=3600):
    # Background loop:
    # 1) Wait for wake word (or timed agent capture)
    # 2) Check voice liveness (if needed)
    # 3) Record short command and execute it
    """Listens for Porcupine wake word and then handles auth + command."""
    session_valid_until = 0
    if not ACCESS_KEY:
        status_queue.put("Porcupine key missing. Wake word disabled.")
        return
    try:
        sens = 0.6
        try:
            s = _load_settings()
            sens = float(s.get("wake_sensitivity", 0.6))
        except Exception:
            pass
        kw = ["hey computer"]
        try:
            s = _load_settings()
            raw = s.get("wake_keywords") or s.get("wake_keyword")
            if raw:
                kw = [k.strip() for k in str(raw).split(',') if k.strip()]
        except Exception:
            pass
        porcupine = pvporcupine.create(
            access_key=ACCESS_KEY,
            keyword_paths=[PORCUPINE_KEYWORD_PATH] if os.path.exists(PORCUPINE_KEYWORD_PATH) else None,
            keywords=kw if not os.path.exists(PORCUPINE_KEYWORD_PATH) else None,
            sensitivities=[sens] if len(kw) <= 1 else [sens] * len(kw)
        )
    except Exception as e:
        print("[Porcupine init error]", e)
        status_queue.put("Porcupine init failed.")
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
                if not agent_active:
                    speak("Yes Master?")
                now = time.time()
                if now > session_valid_until and not agent_active:
                    status_queue.put("Performing liveness check...")
                    if not liveness_check():
                        speak("Access denied. Voice does not match.")
                        status_queue.put("Liveness failed.")
                        continue
                    else:
                        session_valid_until = time.time() + session_duration
                        status_queue.put("Liveness passed. Session active.")
                        speak("Liveness confirmed. Please speak your command after the beep.")

                # respect listening pause
                if time.time() < listening_blocked_until:
                    continue
                # Record user command
                time.sleep(0.25)
                status_queue.put("Recording command...")
                try:
                    winsound.Beep(800, 200)
                except Exception:
                    pass
                # Dynamic recording: 4s min, until user stops speaking
                show_recording_overlay(root_window)
                fn, had = record_until_silence(COMMAND_WAV, on_amp=update_recording_overlay)
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
                    status_queue.put(f"Command: {cmd_text}")
                    try:
                        speak(f"You said: {cmd_text}")
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

def start_gui():
    # Builds a small window with useful buttons and status messages.
    root = tk.Tk()
    global root_window
    root_window = root
    try:
        style = ttk.Style()
        style.theme_use('clam')
    except Exception:
        pass
    root.title("SentinelAI Dashboard")
    root.geometry("420x240")

    status_label = tk.Label(root, text="SentinelAI Running...", font=("Arial", 11))
    status_label.pack(pady=8)

    session_label = tk.Label(root, text="Session: expired", font=("Arial", 10))
    session_label.pack()

    # Diagnostics row: shows whether keys and driver are present
    diag_items = []
    diag_items.append("OpenAI: OK" if os.getenv("OPEN_AI_API_KEY") else "OpenAI: Missing")
    diag_items.append("Porcupine: OK" if os.getenv("PVPORCUPINE_PRIVATE_KEY") else "Porcupine: Missing")
    driver_path = os.path.join(os.path.dirname(__file__), "chromedriver-win64", "chromedriver.exe")
    diag_items.append("Driver: OK" if os.path.exists(driver_path) else "Driver: Missing")
    
    diag_label = tk.Label(root, text=" | ".join(diag_items), font=("Arial", 9))
    diag_label.pack(pady=6)

    register_btn = ttk.Button(
        root,
        text="Register Voice",
        
        command=manual_register_voice
    )
    register_btn.pack(pady=8)

    def refresh_apps():
        status_queue.put("Refreshing apps registry...")
        threading.Thread(target=build_app_registry, daemon=True).start()

    refresh_btn = ttk.Button(
        root,
        text="Refresh Apps Registry",
        
        command=refresh_apps
    )
    refresh_btn.pack(pady=6)

    def show_top_apps():
        items = top_mru(5)
        messagebox.showinfo("Top Apps", "\n".join(items) if items else "No usage yet.")

    top_btn = ttk.Button(
        root,
        text="Show Top Apps",
        command=show_top_apps
    )
    top_btn.pack(pady=6)
    ttk.Button(root, text="Macros", command=lambda: open_macros_ui(root)).pack(pady=6)

    # MRU combobox to open apps quickly
    apps_var = tk.StringVar()
    mru_items = top_mru(10)
    mru_combo = ttk.Combobox(root, textvariable=apps_var, values=mru_items, state='readonly')
    mru_combo.set(mru_items[0] if mru_items else '')
    mru_combo.pack(pady=4, fill='x')

    def open_selected_app():
        name = apps_var.get()
        if not name:
            messagebox.showinfo("Open App", "No app selected.")
            return
        if not open_app(name):
            install_and_open_app(name)

    ttk.Button(root, text="Open Selected App", command=open_selected_app).pack(pady=4)

    def refresh_mru():
        items = top_mru(10)
        mru_combo['values'] = items
        if items:
            mru_combo.set(items[0])

    ttk.Button(root, text="Refresh Suggestions", command=refresh_mru).pack(pady=4)

    def open_settings_ui():
        s = _load_settings()
        win = tk.Toplevel(root)
        win.title("Settings")
        win.geometry("320x260")
        ent_var = tk.BooleanVar(value=entertainment_active)
        auto_var = tk.BooleanVar(value=bool(s.get("autostart", True)))
        bg_var = tk.BooleanVar(value=bool(s.get("background", False)))
        sens_var = tk.DoubleVar(value=float(s.get("wake_sensitivity", 0.6)))

        ttk.Checkbutton(win, text="Entertainment Mode", variable=ent_var).pack(pady=6)
        ttk.Checkbutton(win, text="Autostart on Login", variable=auto_var).pack(pady=6)
        ttk.Checkbutton(win, text="Run in Background (no GUI)", variable=bg_var).pack(pady=6)
        ttk.Label(win, text="Wake Sensitivity").pack()
        ttk.Scale(win, from_=0.1, to=1.0, orient='horizontal', variable=sens_var).pack(fill='x', padx=10)

        def save_settings():
            global entertainment_active
            entertainment_active = ent_var.get()
            data = {
                "autostart": bool(auto_var.get()),
                "background": bool(bg_var.get()),
                "wake_sensitivity": float(sens_var.get())
            }
            _save_settings(data)
            if data["autostart"]:
                ensure_autostart()
            messagebox.showinfo("Settings", "Saved. Some changes apply next start.")
            win.destroy()

        ttk.Button(win, text="Save", command=save_settings).pack(pady=10)

    settings_btn = ttk.Button(
        root,
        text="Settings",
        command=lambda: open_settings_ui_global(root)
    )
    settings_btn.pack(pady=6)

 

    def on_close():
        try:
            start_tray()
            root.withdraw()
            status_queue.put("SentinelAI minimized to tray.")
        except Exception:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    def manual_run_command():
        # Click this to record a command without saying the wake word.
        status_queue.put("Recording command...")
        try:
            winsound.Beep(800, 200)
        except Exception:
            pass
        show_recording_overlay(root)
        fn, had = record_until_silence(COMMAND_WAV, on_amp=update_recording_overlay)
        close_recording_overlay()
        if not had:
            status_queue.put("No speech detected.")
            return
        text = transcribe_wav(fn)
        if not text:
            status_queue.put("Transcription empty.")
            speak("Sorry, I couldn't understand the command.")
        else:
            status_queue.put(f"Command: {text}")
            try:
                speak(f"You said: {text}")
            except Exception:
                pass
            execute_command(text)

    cmd_btn = ttk.Button(
        root,
        text="Record & Execute Command",
        
        command=manual_run_command
    )
    cmd_btn.pack(pady=6)

    agent_start_btn = ttk.Button(
        root,
        text="Start Agent",
        
        command=start_agent
    )
    agent_start_btn.pack(pady=4)

    agent_stop_btn = ttk.Button(
        root,
        text="Stop Agent",
        
        command=stop_agent
    )
    agent_stop_btn.pack(pady=4)

    # Quit button
    quit_button = ttk.Button(root, text="Quit", command=root.destroy)
    quit_button.pack(pady=8)

    # Update UI loop
    def update_ui():
        # Every little while, take messages from the queue and show them on screen
        while not status_queue.empty():
            msg = status_queue.get_nowait()
            status_label.config(text=msg)
        root.after(700, update_ui)

    update_ui()
    root.mainloop()


# ---------- Main ----------
def main():
    # Program start:
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

if __name__ == "__main__":
    main()  # run the program when we start this file
def open_settings_ui_global(parent=None):
    s = _load_settings()
    win = tk.Toplevel(parent) if parent else tk.Tk()
    win.title("Settings")
    win.geometry("320x260")
    ent_var = tk.BooleanVar(value=entertainment_active)
    auto_var = tk.BooleanVar(value=bool(s.get("autostart", True)))
    bg_var = tk.BooleanVar(value=bool(s.get("background", False)))
    sens_var = tk.DoubleVar(value=float(s.get("wake_sensitivity", 0.6)))
    kw_var = tk.StringVar(value=str(s.get("wake_keywords", s.get("wake_keyword", "hey computer"))))
    offline_var = tk.BooleanVar(value=bool(s.get("offline_stt", True)))
    strong_vault_var = tk.BooleanVar(value=bool(s.get("require_strong_vault", False)))

    ttk.Checkbutton(win, text="Entertainment Mode", variable=ent_var).pack(pady=6)
    ttk.Checkbutton(win, text="Autostart on Login", variable=auto_var).pack(pady=6)
    ttk.Checkbutton(win, text="Run in Background (no GUI)", variable=bg_var).pack(pady=6)
    ttk.Label(win, text="Wake Sensitivity").pack()
    ttk.Scale(win, from_=0.1, to=1.0, orient='horizontal', variable=sens_var).pack(fill='x', padx=10)
    ttk.Label(win, text="Wake Keywords (comma-separated)").pack(pady=6)
    ttk.Entry(win, textvariable=kw_var).pack(fill='x', padx=10)
    ttk.Checkbutton(win, text="Enable offline STT fallback", variable=offline_var).pack(pady=6)
    ttk.Checkbutton(win, text="Require strong vault (AES-GCM)", variable=strong_vault_var).pack(pady=6)

    def save_settings():
        global entertainment_active
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
        if data["autostart"]:
            ensure_autostart()
        try:
            restart_wake_listener()
        except Exception:
            pass
        messagebox.showinfo("Settings", "Saved. Wake sensitivity applied. Other changes may apply next start.")
        win.destroy()

    ttk.Button(win, text="Save", command=save_settings).pack(pady=10)
    ttk.Button(win, text="View Logs", command=lambda: open_diagnostics_window(win)).pack(pady=6)

def open_macros_ui(parent=None):
    macros = _macros_load()
    win = tk.Toplevel(parent) if parent else tk.Tk()
    win.title("Macros")
    win.geometry("480x360")
    lst = tk.Listbox(win)
    lst.pack(fill='both', expand=True)
    for k in macros.keys():
        lst.insert('end', k)
    steps_txt = tk.Text(win, height=6)
    steps_txt.pack(fill='x')
    def show_steps(evt=None):
        sel = lst.curselection()
        if not sel:
            return
        name = lst.get(sel[0])
        steps = macros.get(_normalize_name(name), [])
        steps_txt.delete('1.0', 'end')
        steps_txt.insert('1.0', "\n".join(steps))
    lst.bind('<<ListboxSelect>>', show_steps)
    def run_sel():
        sel = lst.curselection()
        if not sel:
            return
        name = lst.get(sel[0])
        steps = macros.get(_normalize_name(name), [])
        speak(f"Running macro {name}")
        def run_steps():
            for s in steps:
                try:
                    execute_command(s)
                    time.sleep(0.8)
                except Exception:
                    pass
        threading.Thread(target=run_steps, daemon=True).start()
    def delete_sel():
        sel = lst.curselection()
        if not sel:
            return
        name = lst.get(sel[0])
        m = _macros_load()
        if m.pop(_normalize_name(name), None) is not None and _macros_save(m):
            speak("Macro deleted")
            lst.delete(sel[0])
            steps_txt.delete('1.0', 'end')
    ttk.Button(win, text="Run", command=run_sel).pack(side='left', padx=8, pady=6)
    ttk.Button(win, text="Delete", command=delete_sel).pack(side='left', padx=8, pady=6)
    name_var = tk.StringVar()
    steps_var = tk.StringVar()
    ttk.Label(win, text="Name").pack(pady=4)
    ttk.Entry(win, textvariable=name_var).pack(fill='x', padx=8)
    ttk.Label(win, text="Steps (semicolon-separated)").pack(pady=4)
    ttk.Entry(win, textvariable=steps_var).pack(fill='x', padx=8)
    def save_new():
        name = name_var.get().strip()
        steps = [s.strip() for s in steps_var.get().split(';') if s.strip()]
        if not name or not steps:
            return
        m = _macros_load()
        m[_normalize_name(name)] = steps
        if _macros_save(m):
            speak("Macro saved")
            lst.insert('end', name)
    ttk.Button(win, text="Save", command=save_new).pack(pady=6)

def open_diagnostics_window(parent=None):
    try:
        lp = os.path.join(APPDATA_DIR, "sentinel.log")
        text = ""
        if os.path.exists(lp):
            with open(lp, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[-400:]
                text = "".join(lines)
        lv = tk.Toplevel(parent) if parent else tk.Tk()
        lv.title("Diagnostics")
        lv.geometry("620x420")
        frm = ttk.Frame(lv)
        frm.pack(fill='both', expand=True)
        txt = tk.Text(frm, wrap='none', height=14)
        txt.insert('1.0', text or "No logs.")
        txt.configure(state='disabled')
        txt.pack(fill='x')
        ttk.Label(frm, text="Mic level").pack(pady=6)
        level_var = tk.DoubleVar(value=0.0)
        bar = ttk.Progressbar(frm, orient='horizontal', mode='determinate', maximum=1.0, variable=level_var)
        bar.pack(fill='x', padx=10)

        def update_level():
            try:
                pa = pyaudio.PyAudio()
                stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=512)
                while True:
                    data = stream.read(512, exception_on_overflow=False)
                    samples = np.frombuffer(data, dtype=np.int16)
                    amp = float(np.mean(np.abs(samples))) / 32768.0
                    level_var.set(min(1.0, amp * 8.0))
                    time.sleep(0.05)
            except Exception:
                level_var.set(0.0)
        threading.Thread(target=update_level, daemon=True).start()
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

def _notes_forget(idx):
    try:
        if os.path.exists(NOTES_PATH):
            with open(NOTES_PATH, "r", encoding="utf-8") as f:
                notes = json.load(f)
            if 1 <= idx <= len(notes):
                notes.pop(idx-1)
                with open(NOTES_PATH, "w", encoding="utf-8") as f:
                    json.dump(notes, f, indent=2)
                return True
    except Exception:
        return False
    return False
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
