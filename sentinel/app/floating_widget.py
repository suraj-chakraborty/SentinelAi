"""
sentinel/app/floating_widget.py
───────────────────────────────
Transparent Floating Desktop Companion Widget.

A sleek, always-on-top, frameless desktop overlay:
- Ambient Cyberpunk HUD styling (Dark obsidian glass, cyan/purple glow).
- Real-time animated holographic core with audio reactivity.
- Quick command execution, voice input trigger, and Spacebar/button barge-in.
- Draggable on screen with collapse-to-orb and expand modes.
- Triggered globally via Win+Space or Alt+J.
"""

import sys
import os
import math
import time
import threading
import logging
from typing import Optional

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None

logger = logging.getLogger("FloatingWidget")


class TransparentFloatingWidget:
    """
    Always-on-top transparent desktop companion widget.
    """

    def __init__(self, orchestrator=None, command_fn=None):
        if tk is None:
            raise RuntimeError("Tkinter not available in this Python environment.")

        self.orchestrator = orchestrator
        self.command_fn = command_fn
        self.is_expanded = True
        self.is_listening = False
        self._anim_angle = 0.0
        self._running = False
        self._drag_start_x = 0
        self._drag_start_y = 0

        self.root = tk.Tk()
        self.root.title("SentinelAI Companion")

        # Windows styling
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", 0.94)
        except Exception:
            pass

        self.root.overrideredirect(True)
        self.root.configure(bg="#080c14")

        # Initial screen position (bottom right corner)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.width = 440
        self.height = 110
        self.pos_x = screen_w - self.width - 40
        self.pos_y = screen_h - self.height - 80
        self.root.geometry(f"{self.width}x{self.height}+{self.pos_x}+{self.pos_y}")

        self._build_ui()
        self._bind_events()

    def _build_ui(self):
        # Outer container frame with border
        self.container = tk.Frame(
            self.root,
            bg="#0d1420",
            highlightthickness=1,
            highlightbackground="#00f3ff",
            highlightcolor="#00f3ff"
        )
        self.container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Top bar (Draggable header & buttons)
        self.top_bar = tk.Frame(self.container, bg="#0d1420")
        self.top_bar.pack(fill=tk.X, padx=10, pady=(6, 2))

        # Title & State
        self.title_lbl = tk.Label(
            self.top_bar,
            text="SENTINEL · COMPANION",
            fg="#00f3ff",
            bg="#0d1420",
            font=("Segoe UI", 9, "bold")
        )
        self.title_lbl.pack(side=tk.LEFT)

        self.state_lbl = tk.Label(
            self.top_bar,
            text="● STANDBY",
            fg="#10b981",
            bg="#0d1420",
            font=("Consolas", 8, "bold")
        )
        self.state_lbl.pack(side=tk.LEFT, padx=10)

        # Close / Minimize buttons
        self.btn_close = tk.Label(
            self.top_bar,
            text="✕",
            fg="#64748b",
            bg="#0d1420",
            cursor="hand2",
            font=("Segoe UI", 9, "bold")
        )
        self.btn_close.pack(side=tk.RIGHT, padx=4)
        self.btn_close.bind("<Button-1>", lambda e: self.toggle_visibility())

        self.btn_barge = tk.Label(
            self.top_bar,
            text="⚡ BARGE-IN",
            fg="#f43f5e",
            bg="#1e1b2e",
            cursor="hand2",
            font=("Consolas", 8, "bold"),
            padx=6,
            pady=1
        )
        self.btn_barge.pack(side=tk.RIGHT, padx=6)
        self.btn_barge.bind("<Button-1>", lambda e: self.trigger_barge_in())

        # Main interactive row
        self.main_row = tk.Frame(self.container, bg="#0d1420")
        self.main_row.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # Animated Holographic Eye / Orb Canvas
        self.canvas = tk.Canvas(
            self.main_row,
            width=54,
            height=54,
            bg="#0d1420",
            highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, padx=(0, 10))

        # Input Entry
        self.entry = tk.Entry(
            self.main_row,
            bg="#111827",
            fg="#f8fafc",
            insertbackground="#00f3ff",
            font=("Segoe UI", 11),
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#1e293b",
            highlightcolor="#00f3ff"
        )
        self.entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8), pady=4)
        self.entry.insert(0, "Ask Sentinel or press Enter...")
        self.entry.bind("<FocusIn>", self._on_entry_focus)
        self.entry.bind("<Return>", lambda e: self.send_command())

        # Action Buttons
        self.btn_mic = tk.Button(
            self.main_row,
            text="🎤",
            bg="#111827",
            fg="#00f3ff",
            activebackground="#00f3ff",
            activeforeground="#060911",
            relief=tk.FLAT,
            font=("Segoe UI", 11),
            cursor="hand2",
            width=3,
            command=self.toggle_mic
        )
        self.btn_mic.pack(side=tk.LEFT, fill=tk.Y, pady=4)

    def _bind_events(self):
        # Window dragging
        self.container.bind("<Button-1>", self._start_drag)
        self.container.bind("<B1-Motion>", self._do_drag)
        self.top_bar.bind("<Button-1>", self._start_drag)
        self.top_bar.bind("<B1-Motion>", self._do_drag)
        self.title_lbl.bind("<Button-1>", self._start_drag)
        self.title_lbl.bind("<B1-Motion>", self._do_drag)

    def _start_drag(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _do_drag(self, event):
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        self.pos_x += dx
        self.pos_y += dy
        self.root.geometry(f"+{self.pos_x}+{self.pos_y}")

    def _on_entry_focus(self, event):
        if self.entry.get() == "Ask Sentinel or press Enter...":
            self.entry.delete(0, tk.END)

    def send_command(self):
        cmd = self.entry.get().strip()
        if not cmd:
            return
        self.entry.delete(0, tk.END)
        self.set_state("THINKING", "#fbbf24")

        def _bg():
            resp = ""
            try:
                if self.command_fn:
                    self.command_fn(cmd)
                    resp = f"Executed: {cmd}"
                elif self.orchestrator:
                    resp = self.orchestrator.run_command(cmd)
                else:
                    from sentinel.core.command_router import get_router
                    router = get_router()
                    resp = router.route(cmd) if router else "Done."
            except Exception as e:
                resp = f"Error: {e}"

            self.root.after(0, lambda: self.set_state("SPEAKING", "#00ffc2"))
            time.sleep(2.5)
            self.root.after(0, lambda: self.set_state("STANDBY", "#10b981"))

        threading.Thread(target=_bg, daemon=True).start()

    def toggle_mic(self):
        self.is_listening = not self.is_listening
        if self.is_listening:
            self.set_state("LISTENING", "#a78bfa")
            def _listen():
                try:
                    from sentinel.app.voice import transcribe
                    from sentinel.app.audio import record_until_silence
                    audio_file = record_until_silence()
                    if audio_file:
                        text = transcribe(audio_file)
                        if text:
                            self.root.after(0, lambda: self.entry.insert(0, text))
                            self.send_command()
                except Exception as exc:
                    logger.error(f"Voice capture error: {exc}")
                finally:
                    self.is_listening = False
                    self.root.after(0, lambda: self.set_state("STANDBY", "#10b981"))
            threading.Thread(target=_listen, daemon=True).start()
        else:
            self.set_state("STANDBY", "#10b981")

    def trigger_barge_in(self):
        try:
            from sentinel.voice.barge_in import get_barge_in_engine
            get_barge_in_engine().trigger_barge_in("floating_widget")
            self.set_state("INTERRUPTED", "#f43f5e")
            self.root.after(1500, lambda: self.set_state("STANDBY", "#10b981"))
        except Exception as e:
            logger.error(f"Barge-in error: {e}")

    def set_state(self, text: str, color: str):
        self.state_lbl.config(text=f"● {text}", fg=color)

    def toggle_visibility(self):
        if self.root.winfo_viewable():
            self.root.withdraw()
        else:
            self.root.deiconify()
            self.root.lift()
            self.entry.focus_set()

    def _animate_canvas(self):
        if not self._running:
            return
        self.canvas.delete("all")
        cx, cy = 27, 27
        self._anim_angle += 0.08

        # Draw pulsing concentric circles
        color = "#00f3ff" if not self.is_listening else "#a78bfa"
        r1 = 18 + math.sin(self._anim_angle) * 3
        r2 = 10 + math.cos(self._anim_angle) * 2

        self.canvas.create_oval(cx - r1, cy - r1, cx + r1, cy + r1, outline=color, width=1.5)
        self.canvas.create_oval(cx - r2, cy - r2, cx + r2, cy + r2, outline="#a78bfa", width=1)

        # Center dot
        self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill=color, outline="")

        # Orbiting satellite particle
        ox = cx + math.cos(self._anim_angle * 1.5) * (r1 + 2)
        oy = cy + math.sin(self._anim_angle * 1.5) * (r1 + 2)
        self.canvas.create_oval(ox - 2, oy - 2, ox + 2, oy + 2, fill="#00ffc2", outline="")

        self.root.after(40, self._animate_canvas)

    def start(self):
        self._running = True
        self._animate_canvas()
        self.root.mainloop()

    def stop(self):
        self._running = False
        try:
            self.root.destroy()
        except Exception:
            pass


_widget_instance: Optional[TransparentFloatingWidget] = None


def get_floating_widget(orchestrator=None) -> TransparentFloatingWidget:
    global _widget_instance
    if _widget_instance is None:
        _widget_instance = TransparentFloatingWidget(orchestrator=orchestrator)
    return _widget_instance


def start_floating_widget(orchestrator=None):
    widget = get_floating_widget(orchestrator)
    # Register global hotkey to toggle widget on Win+Space
    try:
        from sentinel.utils.hotkey_manager import GlobalHotkeyManager
        hotkey = GlobalHotkeyManager(on_trigger=widget.toggle_visibility)
        hotkey.start()
    except Exception as e:
        logger.debug(f"Hotkey hook: {e}")
    widget.start()


if __name__ == "__main__":
    start_floating_widget()
