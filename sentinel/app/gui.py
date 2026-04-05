"""
sentinel/app/gui.py
────────────────────
Cyberpunk Glassmorphism Dashboard and Tray.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pystray
from PIL import Image, ImageDraw
import threading
import os
import logging
from typing import Callable, Optional, Tuple

# App config
from sentinel.app.config import (
    MODERN_BG, MODERN_SURFACE, MODERN_ACCENT, MODERN_GLOW,
    MODERN_TEXT, MODERN_TEXT_MUTED, MODERN_SUCCESS, MODERN_WARNING,
    MODERN_DANGER, MODERN_FONT, MODERN_FONT_BOLD, MODERN_FONT_LARGE
)

logger = logging.getLogger("SentinelGUI")

# ── Styles ──────────────────────────────────────────────────────────────────

def apply_modern_styles(root: tk.Tk):
    """Apply the Modern Light 'Jarvis' theme to a Tk root."""
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass

    root.configure(bg=MODERN_BG)
    # White themes benefit from higher opacity
    root.attributes("-alpha", 1.0) 

    # Frame styles
    style.configure("Glass.TFrame", background=MODERN_SURFACE, relief="flat", borderwidth=0)
    
    # Label Styles
    style.configure("Modern.TLabel", background=MODERN_BG, foreground=MODERN_TEXT, font=MODERN_FONT)
    style.configure("ModernMuted.TLabel", background=MODERN_BG, foreground=MODERN_TEXT_MUTED, font=MODERN_FONT)
    style.configure("ModernHeader.TLabel", background=MODERN_BG, foreground=MODERN_ACCENT, font=MODERN_FONT_LARGE)
    style.configure("Surface.TLabel", background=MODERN_SURFACE, foreground=MODERN_TEXT, font=MODERN_FONT)

    # Button Styles
    style.configure("Modern.TButton", 
                   padding=(15, 8), 
                   relief="flat", 
                   background="#E2E8F0", # Slate 200
                   foreground=MODERN_TEXT,
                   font=MODERN_FONT_BOLD)
    style.map("Modern.TButton",
              background=[('active', MODERN_ACCENT)],
              foreground=[('active', "#FFFFFF")])

    style.configure("ModernAccent.TButton", 
                   padding=(15, 8), 
                   relief="flat", 
                   background=MODERN_ACCENT, 
                   foreground="#FFFFFF",
                   font=MODERN_FONT_BOLD)

    # Progressbar
    style.configure("Modern.Horizontal.TProgressbar", 
                   troughcolor=MODERN_SURFACE, 
                   background=MODERN_ACCENT, 
                   thickness=8,
                   borderwidth=0)

    # Combobox
    style.configure("Modern.TCombobox", 
                   fieldbackground=MODERN_BG, 
                   background=MODERN_BG, 
                   foreground=MODERN_TEXT,
                   arrowcolor=MODERN_ACCENT,
                   font=MODERN_FONT)

# ── Dynamic Elements ─────────────────────────────────────────────────────────

class RecordingOverlay:
    """Always-on-top waveform visualizer."""
    def __init__(self, parent: Optional[tk.Tk] = None):
        self.win = tk.Toplevel(parent) if parent else tk.Tk()
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-alpha", 0.95)
        self.win.configure(bg="#FFFFFF")
        
        # Center the window
        w, h = 340, 110
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.win.geometry(f"{w}x{h}+{x}+{y}")
        
        # UI
        self.frame = tk.Frame(self.win, bg="#FFFFFF", bd=0)
        self.frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        self.lbl = tk.Label(self.frame, text="Sentinel Listening...", 
                           fg=MODERN_ACCENT, bg="#FFFFFF", 
                           font=("Segoe UI Variable Display Semibold", 11))
        self.lbl.pack(pady=(15, 5))
        
        self.canvas = tk.Canvas(self.frame, width=280, height=40, 
                               bg=MODERN_SURFACE, highlightthickness=0)
        self.canvas.pack(padx=30, pady=5)
        self.canvas.create_line(0, 20, 280, 20, fill="#E2E8F0", width=1, tags="base")
        
        self.wave_values = []
        self.active = True

    def update_amp(self, amp: float):
        """Update waveform based on current amplitude."""
        if not self.active or not self.win.winfo_exists(): return
        
        # Normalize
        a = max(0.01, min(1.0, float(amp) * 10.0))
        self.wave_values.append(a)
        if len(self.wave_values) > 50:
            self.wave_values = self.wave_values[-50:]
            
        self.canvas.delete("wave")
        w, h = 280, 40
        mid_y = h // 2
        spacing = w / 50
        
        for i, v in enumerate(self.wave_values):
            x = int(i * spacing)
            offset = int(v * (h/2.5))
            self.canvas.create_line(x, mid_y - offset, x, mid_y + offset, 
                                   fill=MODERN_ACCENT, width=2, tags="wave", capstyle="round")
        
        self.win.update_idletasks()
        self.win.update()

    def destroy(self):
        self.active = False
        try: self.win.destroy()
        except Exception: pass

class ConfigWindow:
    """Settings management window for Voice and System preferences."""
    def __init__(self, parent: tk.Tk):
        from sentinel.app.config import load_settings, save_settings
        self.settings = load_settings()
        self.parent = parent
        
        self.win = tk.Toplevel(parent)
        self.win.title("SENTINEL_CONFIGURATION")
        self.win.geometry("400x500")
        self.win.configure(bg=MODERN_BG)
        self.win.transient(parent)
        self.win.grab_set()

        main = tk.Frame(self.win, bg=MODERN_BG, padx=25, pady=25)
        main.pack(fill='both', expand=True)

        tk.Label(main, text="PREFERENCES", font=MODERN_FONT_LARGE, bg=MODERN_BG, fg=MODERN_TEXT).pack(anchor='w', pady=(0, 20))

        # Voice Selection
        tk.Label(main, text="NEURAL_VOICE_ENGINE", font=MODERN_FONT_BOLD, bg=MODERN_BG, fg=MODERN_ACCENT).pack(anchor='w')
        self.voice_var = tk.StringVar(value=self.settings.get("voice", "en-US-GuyNeural"))
        
        # Common edge-tts voices
        voices = ["en-US-GuyNeural", "en-US-JennyNeural", "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-IN-NeerjaNeural"]
        v_combo = ttk.Combobox(main, textvariable=self.voice_var, values=voices, state='readonly', style="Modern.TCombobox")
        v_combo.pack(fill='x', pady=(5, 20))

        # Wake Word Path
        tk.Label(main, text="CUSTOM_WAKE_WORD_PPN", font=MODERN_FONT_BOLD, bg=MODERN_BG, fg=MODERN_ACCENT).pack(anchor='w')
        self.wake_var = tk.StringVar(value=self.settings.get("wake_word_path", ""))
        p_frame = tk.Frame(main, bg=MODERN_BG)
        p_frame.pack(fill='x', pady=5)
        tk.Entry(p_frame, textvariable=self.wake_var, font=MODERN_FONT, bg=MODERN_SURFACE, fg=MODERN_TEXT, bd=0).pack(side='left', fill='x', expand=True, ipady=4)
        
        def browse():
            from tkinter import filedialog
            path = filedialog.askopenfilename(filetypes=[("Porcupine Model", "*.ppn")])
            if path: self.wake_var.set(path)
        
        ttk.Button(p_frame, text="...", width=3, command=browse).pack(side='right', padx=(5,0))
        tk.Label(main, text="Full path to your .ppn file", font=("Segoe UI Variable Display", 8), bg=MODERN_BG, fg=MODERN_TEXT_MUTED).pack(anchor='w', pady=(0, 20))

        # ID Code
        tk.Label(main, text="IDENTIFICATION_CODE", font=MODERN_FONT_BOLD, bg=MODERN_BG, fg=MODERN_ACCENT).pack(anchor='w')
        self.id_var = tk.StringVar(value=os.getenv("SENTINEL_ID_CODE", "1234"))
        tk.Entry(main, textvariable=self.id_var, font=MODERN_FONT, bg=MODERN_SURFACE, fg=MODERN_TEXT, bd=0).pack(fill='x', pady=5, ipady=4)
        
        # Meter Sensitivity
        tk.Label(main, text="VOICE_METER_SENSITIVITY", font=MODERN_FONT_BOLD, bg=MODERN_BG, fg=MODERN_ACCENT).pack(anchor='w', pady=(15, 0))
        self.sens_var = tk.IntVar(value=self.settings.get("meter_sensitivity", 50))
        tk.Scale(main, variable=self.sens_var, from_=1, to=200, orient='horizontal', bg=MODERN_BG, highlightthickness=0, fg=MODERN_TEXT).pack(fill='x', pady=(0, 20))

        # Save
        def save():
            self.settings["voice"] = self.voice_var.get()
            self.settings["wake_word_path"] = self.wake_var.get()
            self.settings["meter_sensitivity"] = self.sens_var.get()
            os.environ["SENTINEL_ID_CODE"] = self.id_var.get()
            save_settings(self.settings)
            messagebox.showinfo("Config", "Settings saved. Please restart Sentinel for changes to take effect.")
            self.win.destroy()

        ttk.Button(main, text="SAVE_CHANGES", style="ModernAccent.TButton", command=save).pack(fill='x', side='bottom')

# ── System Tray ─────────────────────────────────────────────────────────────

def _make_tray_image():
    """Build a simple 64x64 SA icon."""
    img = Image.new('RGB', (64, 64), color=(30, 30, 30))
    d = ImageDraw.Draw(img)
    d.ellipse((8,8,56,56), outline=(0,200,255), width=3)
    d.text((18,24), 'SA', fill=(255,255,255))
    return img

def start_tray_icon(on_start: Callable, 
                    on_stop: Callable, 
                    on_record: Callable, 
                    on_quit: Callable):
    """Run the Pystray icon in a background thread with stderr suppression."""
    menu = pystray.Menu(
        pystray.MenuItem('Start Agent', on_start),
        pystray.MenuItem('Stop Agent', on_stop),
        pystray.MenuItem('Record Command', on_record),
        pystray.MenuItem('Quit', on_quit)
    )
    icon = pystray.Icon('SentinelAI', _make_tray_image(), 'SentinelAI', menu)
    
    def _run_silently():
        import sys, os
        # Silence stderr for this thread to prevent WNDPROC clutter
        _stderr = sys.stderr
        try:
            with open(os.devnull, 'w') as f:
                sys.stderr = f
                icon.run()
        except Exception:
            pass
        finally:
            sys.stderr = _stderr

    threading.Thread(target=_run_silently, daemon=True).start()
    return icon

# ── Main Dashboard ──────────────────────────────────────────────────────────

def build_dashboard(on_voice_reg: Callable,
                    on_rebuild: Callable,
                    on_manual: Callable,
                    on_config: Callable,
                    on_terminate: Callable,
                    on_agent_toggle: Callable,
                    app_list: list) -> Tuple[tk.Tk, ttk.Progressbar, ttk.Combobox]:
    """Build the responsive modern white Sentinel dashboard."""
    root = tk.Tk()
    apply_modern_styles(root)
    root.title("SENTINEL_OS_V4")
    root.minsize(480, 700)
    
    # Configure root for responsiveness
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    main_frame = tk.Frame(root, bg=MODERN_BG, padx=30, pady=30)
    main_frame.grid(row=0, column=0, sticky='nsew')
    main_frame.columnconfigure(0, weight=1)
    
    # Weights for main frame sections
    for i in range(7):
        main_frame.rowconfigure(i, weight=0)
    # The control frame and footer area can have some expansion
    main_frame.rowconfigure(5, weight=1)

    row = 0
    # Header
    header = tk.Frame(main_frame, bg=MODERN_BG)
    header.grid(row=row, column=0, sticky='ew', pady=(0, 30))
    header.columnconfigure(0, weight=1)
    tk.Label(header, text="SENTINEL_CORE_V4", font=MODERN_FONT_LARGE, bg=MODERN_BG, fg=MODERN_TEXT).grid(row=0, column=0, sticky='w')
    
    # Session Status
    status_frame = tk.Frame(header, bg=MODERN_BG)
    status_frame.grid(row=0, column=1, sticky='e')
    tk.Label(status_frame, text="●", font=MODERN_FONT, bg=MODERN_BG, fg=MODERN_SUCCESS).pack(side='left', padx=5)
    tk.Label(status_frame, text="ACTIVE_SESSION", font=MODERN_FONT, bg=MODERN_BG, fg=MODERN_TEXT_MUTED).pack(side='left')

    row += 1
    # Security Status
    sec_frame = ttk.Frame(main_frame, style="Glass.TFrame")
    sec_frame.grid(row=row, column=0, sticky='ew', pady=(0, 20))
    sec_content = tk.Frame(sec_frame, bg=MODERN_SURFACE, padx=25, pady=20)
    sec_content.pack(fill='both', expand=True)
    tk.Label(sec_content, text="[ IDENTITY_VERIFIED ]: SECURE", font=MODERN_FONT_BOLD, bg=MODERN_SURFACE, fg=MODERN_ACCENT).pack(anchor='w')
    tk.Label(sec_content, text="Session resets at 12:30 AM", font=("Segoe UI Variable Display", 9), bg=MODERN_SURFACE, fg=MODERN_TEXT_MUTED).pack(anchor='w', pady=(4,0))
    
    row += 1
    # Voice Input Meter
    meter_frame = tk.Frame(main_frame, bg=MODERN_BG)
    meter_frame.grid(row=row, column=0, sticky='ew', pady=(0, 30))
    tk.Label(meter_frame, text="AUDIO_INPUT_STREAM", font=("Segoe UI Variable Display", 8, "bold"), bg=MODERN_BG, fg=MODERN_TEXT_MUTED).pack(anchor='w', pady=(0, 5))
    voice_meter = ttk.Progressbar(meter_frame, orient="horizontal", mode="determinate", maximum=100, style="Modern.Horizontal.TProgressbar")
    voice_meter.pack(fill='x')

    row += 1
    # Features / Actions 
    actions = tk.Frame(main_frame, bg=MODERN_BG)
    actions.grid(row=row, column=0, sticky='ew', pady=(0, 30))
    actions.columnconfigure((0,1), weight=1)
    ttk.Button(actions, text="REGISTER_VOICE", style="Modern.TButton", command=on_voice_reg).grid(row=0, column=0, sticky='ew', padx=(0,10))
    ttk.Button(actions, text="REBUILD_INDEX", style="Modern.TButton", command=on_rebuild).grid(row=0, column=1, sticky='ew', padx=(10,0))

    row += 1
    # System Control Logic
    ctrl_frame = ttk.Frame(main_frame, style="Glass.TFrame")
    ctrl_frame.grid(row=row, column=0, sticky='nsew', pady=(0, 30))
    ctrl_content = tk.Frame(ctrl_frame, bg=MODERN_SURFACE, padx=25, pady=25)
    ctrl_content.pack(fill='both', expand=True)
    tk.Label(ctrl_content, text="SYSTEM_HANDLERS", font=MODERN_FONT_BOLD, bg=MODERN_SURFACE, fg=MODERN_TEXT).pack(anchor='w', pady=(0,15))
    
    apps_var = tk.StringVar()
    mru_combo = ttk.Combobox(ctrl_content, textvariable=apps_var, values=app_list, state='readonly', style="Modern.TCombobox")
    mru_combo.pack(fill='x', pady=(0,15))
    
    row += 1
    # Controls
    footer = tk.Frame(main_frame, bg=MODERN_BG)
    footer.grid(row=row, column=0, sticky='ew')
    footer.columnconfigure(0, weight=1)
    
    ttk.Button(footer, text="[ INITIATE_MANUAL_OVERRIDE ]", style="ModernAccent.TButton", command=on_manual).grid(row=0, column=0, sticky='ew', pady=(0,15))
    
    ctrls = tk.Frame(footer, bg=MODERN_BG)
    ctrls.grid(row=1, column=0, sticky='ew')
    ctrls.columnconfigure((0,1), weight=1)
    
    def open_config():
        ConfigWindow(root)

    ttk.Button(ctrls, text="CONFIG", style="Modern.TButton", command=open_config).grid(row=0, column=0, sticky='ew', padx=(0,5))
    ttk.Button(ctrls, text="TERMINATE", style="Modern.TButton", command=on_terminate).grid(row=0, column=1, sticky='ew', padx=(5,0))

    return root, voice_meter, mru_combo
