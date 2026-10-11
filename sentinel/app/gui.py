"""
sentinel/app/gui.py
────────────────────
Cyberpunk / JARVIS Holographic Dark Glassmorphism Dashboard & Tray.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import sys
import math
import random
import logging
from typing import Callable, Optional, Tuple, List

try:
    import pystray
    _PYSTRAY_AVAILABLE = True
except ImportError:
    pystray = None
    _PYSTRAY_AVAILABLE = False

try:
    from PIL import Image, ImageDraw
    _PIL_AVAILABLE = True
except ImportError:
    Image = None
    ImageDraw = None
    _PIL_AVAILABLE = False

# App config
from sentinel.app.config import (
    MODERN_BG, MODERN_SURFACE, MODERN_SURFACE_ALT, MODERN_SURFACE_BORDER,
    MODERN_ACCENT, MODERN_ACCENT_HOVER, MODERN_ACCENT_PURPLE, MODERN_GLOW,
    MODERN_TEXT, MODERN_TEXT_MUTED, MODERN_SUCCESS, MODERN_WARNING,
    MODERN_DANGER, MODERN_FONT, MODERN_FONT_BOLD, MODERN_FONT_LARGE,
    MODERN_FONT_TITLE, MODERN_FONT_MONO
)

logger = logging.getLogger("SentinelGUI")


# ── Windows 10/11 Dark Titlebar ──────────────────────────────────────────────

def enable_dark_titlebar(window: tk.Tk | tk.Toplevel) -> None:
    """Enables Windows 10/11 immersive dark mode on the native window title bar."""
    if os.name == "nt":
        try:
            import ctypes
            window.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
            if not hwnd:
                hwnd = window.winfo_id()
            for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE
                val = ctypes.c_int(1)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attr, ctypes.byref(val), ctypes.sizeof(val)
                )
        except Exception:
            pass


# ── Global Styles ─────────────────────────────────────────────────────────────

def apply_modern_styles(root: tk.Tk) -> None:
    """Apply the Cyberpunk / JARVIS HUD Dark theme to a Tk root."""
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass

    root.configure(bg=MODERN_BG)
    enable_dark_titlebar(root)

    # Frame styles
    style.configure("Glass.TFrame", background=MODERN_SURFACE, relief="flat", borderwidth=0)
    style.configure("DarkMain.TFrame", background=MODERN_BG, relief="flat", borderwidth=0)

    # Label Styles
    style.configure("Modern.TLabel", background=MODERN_BG, foreground=MODERN_TEXT, font=MODERN_FONT)
    style.configure("ModernMuted.TLabel", background=MODERN_BG, foreground=MODERN_TEXT_MUTED, font=MODERN_FONT)
    style.configure("ModernHeader.TLabel", background=MODERN_BG, foreground=MODERN_ACCENT, font=MODERN_FONT_LARGE)
    style.configure("Surface.TLabel", background=MODERN_SURFACE, foreground=MODERN_TEXT, font=MODERN_FONT)

    # Combobox Styles
    style.configure("Modern.TCombobox",
                    fieldbackground=MODERN_SURFACE,
                    background=MODERN_SURFACE_ALT,
                    foreground=MODERN_TEXT,
                    arrowcolor=MODERN_ACCENT,
                    bordercolor=MODERN_SURFACE_BORDER,
                    lightcolor=MODERN_SURFACE_BORDER,
                    darkcolor=MODERN_SURFACE_BORDER,
                    font=MODERN_FONT)
    style.map("Modern.TCombobox",
              fieldbackground=[('readonly', MODERN_SURFACE)],
              selectbackground=[('readonly', MODERN_SURFACE_ALT)],
              selectforeground=[('readonly', MODERN_TEXT)])


# ── Interactive Dynamic Equalizer Visualizer ──────────────────────────────────

class EqualizerVisualizer(tk.Canvas):
    """
    Futuristic Neon Audio Frequency Spectrum Visualizer.
    Drop-in replacement for ttk.Progressbar with .configure(value=v) / .config(value=v).
    """
    def __init__(self, parent, width: int = 440, height: int = 56, num_bars: int = 28, bg: str = "#0B1120"):
        super().__init__(
            parent, width=width, height=height, bg=bg,
            highlightthickness=1, highlightbackground=MODERN_SURFACE_BORDER,
            relief="flat"
        )
        self.num_bars = num_bars
        self.w = width
        self.h = height
        self._value = 0.0
        self._target_heights = [4.0] * num_bars
        self._current_heights = [4.0] * num_bars
        self._peak_caps = [4.0] * num_bars
        self._idle_phase = 0.0
        self._is_active = True

        self.bind("<Configure>", self._on_resize)
        self.bind("<Destroy>", self._on_destroy)
        self._tick()

    def _on_destroy(self, event=None):
        self._is_active = False

    def _on_resize(self, event):
        if event.width > 10 and event.height > 10:
            self.w = event.width
            self.h = event.height

    def configure(self, **kwargs):
        if "value" in kwargs:
            self.set_value(kwargs["value"])
        for k in ("bg", "background", "width", "height"):
            if k in kwargs:
                super().configure(**{k: kwargs[k]})

    def config(self, **kwargs):
        self.configure(**kwargs)

    def __setitem__(self, key, value):
        if key == "value":
            self.set_value(value)
        else:
            super().__setitem__(key, value)

    def set_value(self, val: float):
        try:
            self._value = max(0.0, min(100.0, float(val)))
        except (ValueError, TypeError):
            self._value = 0.0

    def _tick(self):
        if not self._is_active:
            return

        try:
            if not self.winfo_exists():
                return
        except Exception:
            return

        # Update heights
        max_h = max(10, self.h - 14)
        if self._value > 0.5:
            # Active input from mic
            for i in range(self.num_bars):
                norm_pos = abs(i - self.num_bars / 2) / (self.num_bars / 2)
                bell = math.cos(norm_pos * 1.25)
                jitter = random.uniform(0.7, 1.3)
                target = max(3.0, (self._value / 100.0) * max_h * bell * jitter)
                self._target_heights[i] = min(max_h, target)
            # Smooth decay of input
            self._value *= 0.88
        else:
            # Ambient resting harmonic wave
            self._idle_phase += 0.08
            for i in range(self.num_bars):
                wave = math.sin(self._idle_phase + (i * 0.32))
                self._target_heights[i] = 4.0 + 3.5 * (wave + 1.0) / 2.0

        # Interpolate & update peak caps
        for i in range(self.num_bars):
            self._current_heights[i] += (self._target_heights[i] - self._current_heights[i]) * 0.35
            # Peaks
            if self._current_heights[i] > self._peak_caps[i]:
                self._peak_caps[i] = self._current_heights[i]
            else:
                self._peak_caps[i] = max(self._current_heights[i], self._peak_caps[i] - 0.7)

        # Redraw bars
        self.delete("eq")
        bar_margin = 8
        avail_w = max(50, self.w - (bar_margin * 2))
        bar_pitch = avail_w / self.num_bars
        bar_w = max(2, int(bar_pitch * 0.70))

        y_base = self.h - 6

        # Draw ambient base line
        self.create_line(bar_margin, y_base + 1, self.w - bar_margin, y_base + 1,
                         fill="#1E293B", width=1, tags="eq")

        for i in range(self.num_bars):
            bx = int(bar_margin + (i * bar_pitch))
            bh = int(self._current_heights[i])
            by = y_base - bh

            # Cyan to Violet gradient interpolation
            ratio = i / max(1, self.num_bars - 1)
            # Gradient: (0, 229, 255) -> (139, 92, 246)
            r = int(0 + (139 - 0) * ratio)
            g = int(229 + (92 - 229) * ratio)
            b = int(255 + (246 - 255) * ratio)
            color = f"#{r:02x}{g:02x}{b:02x}"

            # Main frequency bar
            self.create_rectangle(bx, by, bx + bar_w, y_base, fill=color, outline="", tags="eq")

            # Peak cap
            peak_y = int(y_base - self._peak_caps[i]) - 2
            if peak_y > 2:
                self.create_rectangle(bx, peak_y - 1, bx + bar_w, peak_y,
                                      fill="#FFFFFF", outline="", tags="eq")

        self.after(35, self._tick)


# ── Cyberpunk Tech Button ─────────────────────────────────────────────────────

class CyberButton(tk.Canvas):
    """
    High-tech chamfered interactive button with neon borders and hover animations.
    """
    def __init__(
        self, parent, text: str, command: Optional[Callable] = None,
        icon: str = "", subtitle: str = "",
        bg_color: str = "#0F172A", hover_bg: str = "#1E293B",
        border_color: str = "#1E293B", hover_border: str = "#00E5FF",
        text_color: str = "#F8FAFC", accent_color: str = "#00E5FF",
        width: int = 200, height: int = 46, chamfer: int = 7
    ):
        super().__init__(
            parent, width=width, height=height, bg=MODERN_BG,
            highlightthickness=0, relief="flat", cursor="hand2"
        )
        self.text = text
        self.icon = icon
        self.subtitle = subtitle
        self.command = command
        self.w = width
        self.h = height
        self.chamfer = chamfer

        self.bg_color = bg_color
        self.hover_bg = hover_bg
        self.border_color = border_color
        self.hover_border = hover_border
        self.text_color = text_color
        self.accent_color = accent_color

        self._hovered = False
        self._pressed = False

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_down)
        self.bind("<ButtonRelease-1>", self._on_up)
        self.bind("<Configure>", self._on_resize)
        self._draw()

    def _on_resize(self, event):
        if event.width > 10 and event.height > 10:
            self.w = event.width
            self.h = event.height
            self._draw()

    def _on_enter(self, event):
        self._hovered = True
        self._draw()

    def _on_leave(self, event):
        self._hovered = False
        self._pressed = False
        self._draw()

    def _on_down(self, event):
        self._pressed = True
        self._draw()

    def _on_up(self, event):
        if self._pressed and self.command:
            try:
                self.command()
            except Exception as e:
                logger.error(f"Error executing button command: {e}")
        self._pressed = False
        self._draw()

    def _draw(self):
        self.delete("all")
        c = self.chamfer
        x1, y1 = 2, 2
        x2, y2 = self.w - 3, self.h - 3

        fill = self.hover_bg if self._hovered else self.bg_color
        outline = self.hover_border if self._hovered else self.border_color
        if self._pressed:
            fill = "#083344"
            outline = "#38BDF8"

        points = [
            x1 + c, y1,
            x2 - c, y1,
            x2, y1 + c,
            x2, y2 - c,
            x2 - c, y2,
            x1 + c, y2,
            x1, y2 - c,
            x1, y1 + c
        ]
        self.create_polygon(points, fill=fill, outline=outline, width=1)

        # Subtle glowing top accent on hover
        if self._hovered:
            self.create_line(x1 + c, y1 + 1, x2 - c, y1 + 1, fill=self.accent_color, width=2)

        mid_y = self.h // 2
        content_text = f"{self.icon} {self.text}".strip() if self.icon else self.text

        if self.subtitle:
            self.create_text(
                self.w // 2, mid_y - 6,
                text=content_text, fill=self.text_color,
                font=MODERN_FONT_BOLD
            )
            self.create_text(
                self.w // 2, mid_y + 11,
                text=self.subtitle, fill=MODERN_TEXT_MUTED,
                font=("Segoe UI Variable Text", 8)
            )
        else:
            self.create_text(
                self.w // 2, mid_y,
                text=content_text, fill=self.text_color,
                font=MODERN_FONT_BOLD
            )


# ── Hero Voice Command Button ─────────────────────────────────────────────────

class HeroVoiceButton(tk.Canvas):
    """
    Large prominent glowing cyber hero button for voice input.
    """
    def __init__(self, parent, text: str, command: Optional[Callable] = None, height: int = 54):
        super().__init__(
            parent, height=height, bg=MODERN_BG,
            highlightthickness=0, relief="flat", cursor="hand2"
        )
        self.text = text
        self.command = command
        self.h = height
        self.w = 400
        self._hovered = False
        self._pressed = False

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_down)
        self.bind("<ButtonRelease-1>", self._on_up)
        self.bind("<Configure>", self._on_resize)
        self._draw()

    def _on_resize(self, event):
        if event.width > 20:
            self.w = event.width
            self._draw()

    def _on_enter(self, event):
        self._hovered = True
        self._draw()

    def _on_leave(self, event):
        self._hovered = False
        self._pressed = False
        self._draw()

    def _on_down(self, event):
        self._pressed = True
        self._draw()

    def _on_up(self, event):
        if self._pressed and self.command:
            try:
                self.command()
            except Exception as e:
                logger.error(f"Error executing hero button: {e}")
        self._pressed = False
        self._draw()

    def _draw(self):
        self.delete("all")
        c = 10
        x1, y1 = 2, 2
        x2, y2 = self.w - 3, self.h - 3

        bg_fill = "#0E3F5A" if self._hovered else "#082B3E"
        outline = "#00E5FF" if self._hovered else "#0284C7"
        if self._pressed:
            bg_fill = "#0C4A6E"
            outline = "#38BDF8"

        points = [
            x1 + c, y1,
            x2 - c, y1,
            x2, y1 + c,
            x2, y2 - c,
            x2 - c, y2,
            x1 + c, y2,
            x1, y2 - c,
            x1, y1 + c
        ]
        self.create_polygon(points, fill=bg_fill, outline=outline, width=1.5)

        # Electric glowing top highlight
        self.create_line(x1 + c, y1 + 1, x2 - c, y1 + 1, fill="#00E5FF", width=2.5)

        mid_y = self.h // 2
        self.create_text(
            self.w // 2, mid_y - 7,
            text=f"⚡ {self.text}", fill="#F8FAFC",
            font=("Segoe UI Variable Display", 11, "bold")
        )
        self.create_text(
            self.w // 2, mid_y + 11,
            text="CLICK TO SPEAK  •  OR PRESS [ ALT + J ]", fill="#7DD3FC",
            font=("Segoe UI Variable Text", 8, "bold")
        )


# ── Mini Arc-Reactor Emblem ───────────────────────────────────────────────────

def draw_arc_reactor_icon(canvas: tk.Canvas, cx: int = 20, cy: int = 20, r: int = 15):
    """Draws a sleek glowing arc-reactor emblem on a canvas."""
    canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#00E5FF", width=2)
    canvas.create_oval(cx - (r - 4), cy - (r - 4), cx + (r - 4), cy + (r - 4), outline="#0284C7", width=1)
    canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="#00E5FF", outline="")


# ── Dynamic Recording Overlay ─────────────────────────────────────────────────

class RecordingOverlay:
    """Always-on-top waveform visualizer in sleek cyberpunk dark mode."""
    def __init__(self, parent: Optional[tk.Tk] = None):
        self.win = tk.Toplevel(parent) if parent else tk.Tk()
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-alpha", 0.95)
        self.win.configure(bg=MODERN_BG)

        # Center the window
        w, h = 340, 110
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.win.geometry(f"{w}x{h}+{x}+{y}")

        self.frame = tk.Frame(self.win, bg=MODERN_SURFACE, bd=1, highlightbackground=MODERN_ACCENT, highlightthickness=1)
        self.frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.lbl = tk.Label(
            self.frame, text="⚡ Sentinel Listening...",
            fg=MODERN_ACCENT, bg=MODERN_SURFACE,
            font=("Segoe UI Variable Display", 11, "bold")
        )
        self.lbl.pack(pady=(15, 5))

        self.canvas = tk.Canvas(self.frame, width=280, height=36, bg="#0B1120", highlightthickness=0)
        self.canvas.pack(padx=30, pady=5)
        self.canvas.create_line(0, 18, 280, 18, fill=MODERN_SURFACE_BORDER, width=1, tags="base")

        self.wave_values = []
        self.active = True

    def update_amp(self, amp: float):
        """Update waveform based on current amplitude."""
        if not self.active or not self.win.winfo_exists():
            return

        a = max(0.01, min(1.0, float(amp) * 10.0))
        self.wave_values.append(a)
        if len(self.wave_values) > 50:
            self.wave_values = self.wave_values[-50:]

        self.canvas.delete("wave")
        w, h = 280, 36
        mid_y = h // 2
        spacing = w / 50

        for i, v in enumerate(self.wave_values):
            x = int(i * spacing)
            offset = int(v * (h / 2.5))
            self.canvas.create_line(
                x, mid_y - offset, x, mid_y + offset,
                fill=MODERN_ACCENT, width=2, tags="wave", capstyle="round"
            )

        self.win.update_idletasks()
        self.win.update()

    def destroy(self):
        self.active = False
        try:
            self.win.destroy()
        except Exception:
            pass


# ── Settings Configuration Window ─────────────────────────────────────────────

class ConfigWindow:
    """Settings management window with matching dark cyberpunk aesthetics."""
    def __init__(self, parent: tk.Tk):
        from sentinel.app.config import load_settings, save_settings
        self.settings = load_settings()
        self.parent = parent

        self.win = tk.Toplevel(parent)
        self.win.title("SENTINEL // PREFERENCES")
        self.win.geometry("440x540")
        self.win.configure(bg=MODERN_BG)
        enable_dark_titlebar(self.win)
        self.win.transient(parent)
        self.win.grab_set()

        main = tk.Frame(self.win, bg=MODERN_BG, padx=25, pady=25)
        main.pack(fill='both', expand=True)

        tk.Label(
            main, text="SYSTEM PREFERENCES",
            font=MODERN_FONT_LARGE, bg=MODERN_BG, fg=MODERN_TEXT
        ).pack(anchor='w', pady=(0, 20))

        # Voice Selection
        tk.Label(
            main, text="NEURAL VOICE PROFILE",
            font=MODERN_FONT_BOLD, bg=MODERN_BG, fg=MODERN_ACCENT
        ).pack(anchor='w')
        self.voice_var = tk.StringVar(value=self.settings.get("voice", "en-US-JennyNeural"))

        voices = ["en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural", "en-GB-RyanNeural", "en-IN-NeerjaNeural"]
        v_combo = ttk.Combobox(main, textvariable=self.voice_var, values=voices, state='readonly', style="Modern.TCombobox")
        v_combo.pack(fill='x', pady=(6, 18))

        # Wake Word Path
        tk.Label(
            main, text="CUSTOM WAKE WORD (.PPN)",
            font=MODERN_FONT_BOLD, bg=MODERN_BG, fg=MODERN_ACCENT
        ).pack(anchor='w')
        self.wake_var = tk.StringVar(value=self.settings.get("wake_word_path", ""))
        p_frame = tk.Frame(main, bg=MODERN_BG)
        p_frame.pack(fill='x', pady=6)

        entry_wake = tk.Entry(
            p_frame, textvariable=self.wake_var, font=MODERN_FONT,
            bg=MODERN_SURFACE, fg=MODERN_TEXT, insertbackground=MODERN_ACCENT,
            bd=1, highlightbackground=MODERN_SURFACE_BORDER, relief="flat"
        )
        entry_wake.pack(side='left', fill='x', expand=True, ipady=4)

        def browse():
            from tkinter import filedialog
            path = filedialog.askopenfilename(filetypes=[("Porcupine Model", "*.ppn")])
            if path:
                self.wake_var.set(path)

        CyberButton(p_frame, text="...", command=browse, width=42, height=30).pack(side='right', padx=(6, 0))
        tk.Label(
            main, text="Path to your trained Picovoice Porcupine .ppn model",
            font=("Segoe UI Variable Text", 8), bg=MODERN_BG, fg=MODERN_TEXT_MUTED
        ).pack(anchor='w', pady=(0, 18))

        # ID Code
        tk.Label(
            main, text="SECURITY PASSCODE (ID CODE)",
            font=MODERN_FONT_BOLD, bg=MODERN_BG, fg=MODERN_ACCENT
        ).pack(anchor='w')
        self.id_var = tk.StringVar(value=os.getenv("SENTINEL_ID_CODE", "1234"))
        entry_id = tk.Entry(
            main, textvariable=self.id_var, font=MODERN_FONT,
            bg=MODERN_SURFACE, fg=MODERN_TEXT, insertbackground=MODERN_ACCENT,
            bd=1, highlightbackground=MODERN_SURFACE_BORDER, relief="flat"
        )
        entry_id.pack(fill='x', pady=6, ipady=4)

        # Meter Sensitivity
        tk.Label(
            main, text="VOICE METER SENSITIVITY",
            font=MODERN_FONT_BOLD, bg=MODERN_BG, fg=MODERN_ACCENT
        ).pack(anchor='w', pady=(16, 0))
        self.sens_var = tk.IntVar(value=self.settings.get("meter_sensitivity", 50))
        scale = tk.Scale(
            main, variable=self.sens_var, from_=1, to=200, orient='horizontal',
            bg=MODERN_BG, highlightthickness=0, fg=MODERN_TEXT,
            troughcolor=MODERN_SURFACE, activebackground=MODERN_ACCENT
        )
        scale.pack(fill='x', pady=(0, 20))

        # Save Button
        def save():
            self.settings["voice"] = self.voice_var.get()
            self.settings["wake_word_path"] = self.wake_var.get()
            self.settings["meter_sensitivity"] = self.sens_var.get()
            os.environ["SENTINEL_ID_CODE"] = self.id_var.get()
            save_settings(self.settings)
            messagebox.showinfo("Config", "Settings saved successfully.")
            self.win.destroy()

        CyberButton(
            main, text="SAVE PREFERENCES", command=save,
            bg_color="#0E3F5A", hover_bg="#0284C7", border_color="#00E5FF",
            width=380, height=44
        ).pack(fill='x', side='bottom')


# ── System Tray ───────────────────────────────────────────────────────────────

def _make_tray_image():
    """Build a vibrant 64x64 cyberpunk SA icon for the system tray."""
    img = Image.new('RGB', (64, 64), color=(8, 12, 20))
    d = ImageDraw.Draw(img)
    d.ellipse((6, 6, 58, 58), outline=(0, 229, 255), width=3)
    d.ellipse((14, 14, 50, 50), outline=(14, 165, 233), width=1)
    d.text((18, 22), 'SA', fill=(255, 255, 255))
    return img


def start_tray_icon(on_start: Callable,
                    on_stop: Callable,
                    on_record: Callable,
                    on_quit: Callable):
    """Run the Pystray icon in a background thread."""
    if not _PYSTRAY_AVAILABLE or not _PIL_AVAILABLE:
        logger.debug("pystray or PIL not installed; system tray icon disabled.")
        return None

    menu = pystray.Menu(
        pystray.MenuItem('Start Agent', on_start),
        pystray.MenuItem('Stop Agent', on_stop),
        pystray.MenuItem('Record Command', on_record),
        pystray.MenuItem('Quit', on_quit)
    )
    icon = pystray.Icon('SentinelAI', _make_tray_image(), 'SentinelAI', menu)

    def _run_silently():
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


# ── Main Dashboard ───────────────────────────────────────────────────────────

def build_dashboard(
    on_voice_reg: Callable,
    on_rebuild: Callable,
    on_manual: Callable,
    on_config: Callable,
    on_terminate: Callable,
    on_agent_toggle: Callable,
    app_list: list
) -> Tuple[tk.Tk, EqualizerVisualizer, ttk.Combobox]:
    """
    Build the state-of-the-art Cyberpunk / JARVIS HUD Dark Dashboard.
    """
    root = tk.Tk()
    apply_modern_styles(root)
    root.title("SENTINEL_OS_V4 // JARVIS HUD")
    root.minsize(500, 750)
    root.geometry("520x760")

    # Responsive root setup
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Scrollable container or main container
    container = tk.Frame(root, bg=MODERN_BG, padx=26, pady=24)
    container.grid(row=0, column=0, sticky='nsew')
    container.columnconfigure(0, weight=1)

    # 1. HEADER SECTION
    header = tk.Frame(container, bg=MODERN_BG)
    header.grid(row=0, column=0, sticky='ew', pady=(0, 20))
    header.columnconfigure(1, weight=1)

    # Arc-reactor logo
    logo_canvas = tk.Canvas(header, width=38, height=38, bg=MODERN_BG, highlightthickness=0)
    logo_canvas.grid(row=0, column=0, rowspan=2, padx=(0, 12))
    draw_arc_reactor_icon(logo_canvas, 19, 19, 16)

    title_frame = tk.Frame(header, bg=MODERN_BG)
    title_frame.grid(row=0, column=1, sticky='w')
    tk.Label(
        title_frame, text="SENTINEL_CORE_V4",
        font=MODERN_FONT_TITLE, bg=MODERN_BG, fg=MODERN_TEXT
    ).pack(anchor='w')
    tk.Label(
        title_frame, text="AUTONOMOUS NEURAL OVERWATCH",
        font=("Segoe UI Variable Text", 8, "bold"), bg=MODERN_BG, fg=MODERN_ACCENT_HOVER
    ).pack(anchor='w')

    # Status Pill Badge
    pill = tk.Frame(header, bg="#052E16", padx=10, pady=5, highlightbackground="#166534", highlightthickness=1)
    pill.grid(row=0, column=2, sticky='e')
    tk.Label(pill, text="●", font=("Segoe UI", 8), bg="#052E16", fg=MODERN_SUCCESS).pack(side='left', padx=(0, 5))
    tk.Label(pill, text="ACTIVE_SESSION", font=("Segoe UI Variable Text", 8, "bold"), bg="#052E16", fg="#86EFAC").pack(side='left')

    # 2. SECURITY & HUD TELEMETRY CARD
    sec_card = tk.Frame(container, bg=MODERN_SURFACE, highlightbackground=MODERN_SURFACE_BORDER, highlightthickness=1, padx=20, pady=16)
    sec_card.grid(row=1, column=0, sticky='ew', pady=(0, 20))
    sec_card.columnconfigure(0, weight=1)

    # Top highlight line
    sec_top = tk.Frame(sec_card, bg=MODERN_ACCENT, height=2)
    sec_top.pack(fill='x', pady=(0, 12))

    # Security header
    sec_row = tk.Frame(sec_card, bg=MODERN_SURFACE)
    sec_row.pack(fill='x')
    tk.Label(
        sec_row, text="🛡️ [ IDENTITY_VERIFIED ]: SECURE",
        font=MODERN_FONT_BOLD, bg=MODERN_SURFACE, fg=MODERN_ACCENT
    ).pack(side='left')
    tk.Label(
        sec_row, text="256-BIT VAULT",
        font=MODERN_FONT_MONO, bg=MODERN_SURFACE, fg=MODERN_TEXT_MUTED
    ).pack(side='right')

    tk.Label(
        sec_card, text="Cryptographic session authenticated • Auto-refreshes at 12:30 AM",
        font=("Segoe UI Variable Text", 8), bg=MODERN_SURFACE, fg=MODERN_TEXT_MUTED
    ).pack(anchor='w', pady=(3, 12))

    # Mini telemetry chip row
    chips_frame = tk.Frame(sec_card, bg=MODERN_SURFACE)
    chips_frame.pack(fill='x')
    for label_txt, val_txt in [("PERSONA", "JARVIS"), ("SPEECH", "EDGE-NEURAL"), ("HOTKEY", "ALT+J")]:
        c = tk.Frame(chips_frame, bg=MODERN_SURFACE_ALT, padx=8, pady=4, highlightbackground=MODERN_SURFACE_BORDER, highlightthickness=1)
        c.pack(side='left', padx=(0, 8))
        tk.Label(c, text=label_txt, font=("Segoe UI Variable Text", 7, "bold"), bg=MODERN_SURFACE_ALT, fg=MODERN_TEXT_MUTED).pack(side='left', padx=(0, 4))
        tk.Label(c, text=val_txt, font=("Segoe UI Variable Text", 8, "bold"), bg=MODERN_SURFACE_ALT, fg=MODERN_TEXT).pack(side='left')

    # 3. AUDIO INPUT SPECTRUM VISUALIZER
    spectrum_frame = tk.Frame(container, bg=MODERN_BG)
    spectrum_frame.grid(row=2, column=0, sticky='ew', pady=(0, 20))
    spectrum_frame.columnconfigure(0, weight=1)

    spec_hdr = tk.Frame(spectrum_frame, bg=MODERN_BG)
    spec_hdr.pack(fill='x', pady=(0, 6))
    tk.Label(
        spec_hdr, text="AUDIO_INPUT_STREAM",
        font=("Segoe UI Variable Text", 8, "bold"), bg=MODERN_BG, fg=MODERN_TEXT_MUTED
    ).pack(side='left')
    tk.Label(
        spec_hdr, text="SPECTRUM FREQUENCY // 28 BANDS",
        font=MODERN_FONT_MONO, bg=MODERN_BG, fg=MODERN_ACCENT
    ).pack(side='right')

    voice_meter = EqualizerVisualizer(spectrum_frame, height=54, bg="#08101D")
    voice_meter.pack(fill='x')

    # 4. FRONTIER CAPABILITIES ACTIONS GRID (2x2)
    actions_grid = tk.Frame(container, bg=MODERN_BG)
    actions_grid.grid(row=3, column=0, sticky='ew', pady=(0, 20))
    actions_grid.columnconfigure((0, 1), weight=1)

    btn_voice = CyberButton(
        actions_grid, text="REGISTER_VOICE", icon="🎙️",
        subtitle="Voiceprint Calibration", command=on_voice_reg, height=48
    )
    btn_voice.grid(row=0, column=0, sticky='ew', padx=(0, 6), pady=(0, 8))

    btn_index = CyberButton(
        actions_grid, text="REBUILD_INDEX", icon="⚡",
        subtitle="Knowledge Sync", command=on_rebuild, height=48
    )
    btn_index.grid(row=0, column=1, sticky='ew', padx=(6, 0), pady=(0, 8))

    # Helper commands for quick agent toggle / avatar
    def toggle_avatar():
        try:
            from sentinel.ar.holographic_avatar import get_avatar_engine
            av = get_avatar_engine()
            if getattr(av, "is_visible", False):
                av.hide()
            else:
                av.show()
        except Exception:
            pass

    btn_avatar = CyberButton(
        actions_grid, text="HOLO_AVATAR", icon="🔮",
        subtitle="3D Desktop Companion", command=toggle_avatar, height=48
    )
    btn_avatar.grid(row=1, column=0, sticky='ew', padx=(0, 6))

    btn_gestures = CyberButton(
        actions_grid, text="AIR_GESTURES", icon="🖐️",
        subtitle="Spatial Hand Tracking", command=on_agent_toggle, height=48
    )
    btn_gestures.grid(row=1, column=1, sticky='ew', padx=(6, 0))

    # 5. SYSTEM HANDLERS / QUICK LAUNCHER CARD
    ctrl_card = tk.Frame(container, bg=MODERN_SURFACE, highlightbackground=MODERN_SURFACE_BORDER, highlightthickness=1, padx=20, pady=16)
    ctrl_card.grid(row=4, column=0, sticky='ew', pady=(0, 20))
    ctrl_card.columnconfigure(0, weight=1)

    tk.Label(
        ctrl_card, text="SYSTEM_HANDLERS // REGISTERED EXTENSIONS",
        font=MODERN_FONT_BOLD, bg=MODERN_SURFACE, fg=MODERN_TEXT
    ).pack(anchor='w', pady=(0, 8))

    apps_var = tk.StringVar()
    mru_combo = ttk.Combobox(
        ctrl_card, textvariable=apps_var, values=app_list, state='readonly', style="Modern.TCombobox"
    )
    mru_combo.pack(fill='x')

    # 6. HERO VOICE OVERRIDE BUTTON
    hero_btn = HeroVoiceButton(
        container, text="INITIATE VOICE OVERRIDE", command=on_manual, height=54
    )
    hero_btn.grid(row=5, column=0, sticky='ew', pady=(0, 16))

    # 7. FOOTER CONTROLS & STATUS BAR
    footer = tk.Frame(container, bg=MODERN_BG)
    footer.grid(row=6, column=0, sticky='ew')
    footer.columnconfigure((0, 1), weight=1)

    def open_config():
        if on_config and callable(on_config):
            try:
                on_config()
                return
            except Exception:
                pass
        ConfigWindow(root)

    CyberButton(
        footer, text="CONFIG", icon="⚙️", command=open_config,
        bg_color="#111827", hover_bg="#1E293B", height=40
    ).grid(row=0, column=0, sticky='ew', padx=(0, 6), pady=(0, 12))

    CyberButton(
        footer, text="TERMINATE", icon="✕", command=on_terminate,
        bg_color="#1F1523", hover_bg="#451A24", border_color="#7F1D1D",
        hover_border="#EF4444", text_color="#FCA5A5", height=40
    ).grid(row=0, column=1, sticky='ew', padx=(6, 0), pady=(0, 12))

    # Live HUD Status Bar
    status_bar = tk.Frame(footer, bg=MODERN_SURFACE, padx=12, pady=7, highlightbackground=MODERN_SURFACE_BORDER, highlightthickness=1)
    status_bar.grid(row=1, column=0, columnspan=2, sticky='ew')
    status_bar.columnconfigure(1, weight=1)

    tk.Label(status_bar, text="●", font=("Segoe UI", 9), bg=MODERN_SURFACE, fg=MODERN_ACCENT).grid(row=0, column=0, padx=(0, 6))

    status_label = tk.Label(
        status_bar, text="Sentinel Ready. Press Alt+J or Click Voice Override.",
        font=("Segoe UI Variable Text", 9), bg=MODERN_SURFACE, fg=MODERN_TEXT_MUTED, anchor='w'
    )
    status_label.grid(row=0, column=1, sticky='w')

    # Attach references to root
    root.status_label = status_label
    root.voice_meter = voice_meter
    root.mru_combo = mru_combo

    return root, voice_meter, mru_combo
