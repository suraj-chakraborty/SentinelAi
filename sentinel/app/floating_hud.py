"""
sentinel/app/floating_hud.py
────────────────────────────
Ambient Glassmorphism Floating HUD & Audio Visualizer.
Sleek, frameless desktop island that dynamically reflects assistant states.
"""

import logging
import tkinter as tk

logger = logging.getLogger("FloatingHUD")

# Theme Palette (Cyberpunk / Modern Glass)
HUD_BG = "#0B0F19"
HUD_SURFACE = "#1E293B"
HUD_CYAN = "#00F0FF"
HUD_VIOLET = "#A855F7"
HUD_EMERALD = "#10B981"
HUD_TEXT = "#F8FAFC"
HUD_MUTED = "#94A3B8"


class FloatingHUD:
    """
    Always-on-top, frameless floating glass pill.
    Displays dynamic audio waveform and status animations.
    """

    def __init__(self, parent: tk.Tk | None = None):
        self.root = parent or tk.Tk()
        self.win = tk.Toplevel(self.root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-alpha", 0.94)
        self.win.configure(bg=HUD_BG)

        # Center top geometry
        self.width, self.height = 380, 84
        sw = self.win.winfo_screenwidth()
        x = (sw - self.width) // 2
        y = 35  # floating near top edge
        self.win.geometry(f"{self.width}x{self.height}+{x}+{y}")

        # Border / Container
        self.frame = tk.Frame(
            self.win, bg=HUD_BG, bd=1, relief="solid", highlightbackground=HUD_SURFACE, highlightthickness=1
        )
        self.frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Status Label
        self.status_lbl = tk.Label(
            self.frame,
            text="SENTINEL // STANDBY",
            font=("Segoe UI Variable Display Semibold", 10),
            fg=HUD_CYAN,
            bg=HUD_BG,
        )
        self.status_lbl.pack(pady=(10, 2))

        # Audio / Wave Canvas
        self.canvas = tk.Canvas(self.frame, width=320, height=36, bg=HUD_BG, highlightthickness=0)
        self.canvas.pack(pady=(0, 6))

        self.wave_history: list[float] = [0.05] * 40
        self.state = "IDLE"
        self._pulse_phase = 0.0
        self._visible = True

    def set_state(self, state: str, text: str | None = None):
        """Update HUD state: 'IDLE', 'LISTENING', 'PROCESSING', 'SPEAKING'."""
        self.state = state.upper()
        display_text = text or f"SENTINEL // {self.state}"

        color_map = {"IDLE": HUD_MUTED, "LISTENING": HUD_CYAN, "PROCESSING": HUD_VIOLET, "SPEAKING": HUD_EMERALD}
        color = color_map.get(self.state, HUD_CYAN)

        def _update():
            self.status_lbl.configure(text=display_text[:40], fg=color)

        try:
            self.win.after(0, _update)
        except Exception:
            pass

    def update_amp(self, amp: float):
        """Update live audio waveform."""
        if not self._visible or not self.win.winfo_exists():
            return

        normalized = max(0.04, min(1.0, float(amp) * 8.0))
        self.wave_history.append(normalized)
        if len(self.wave_history) > 40:
            self.wave_history.pop(0)

        def _draw():
            try:
                self.canvas.delete("all")
                w, h = 320, 36
                mid_y = h // 2
                spacing = w / 40

                color = (
                    HUD_CYAN if self.state == "LISTENING" else (HUD_EMERALD if self.state == "SPEAKING" else HUD_VIOLET)
                )

                for i, v in enumerate(self.wave_history):
                    x = int(i * spacing)
                    offset = max(2, int(v * (mid_y - 2)))
                    self.canvas.create_line(x, mid_y - offset, x, mid_y + offset, fill=color, width=2, capstyle="round")
            except Exception:
                pass

        try:
            self.win.after(0, _draw)
        except Exception:
            pass

    def hide(self):
        self._visible = False
        try:
            self.win.withdraw()
        except Exception:
            pass

    def show(self):
        self._visible = True
        try:
            self.win.deiconify()
        except Exception:
            pass

    def destroy(self):
        self._visible = False
        try:
            self.win.destroy()
        except Exception:
            pass
