"""
sentinel/core/computer_use_agent.py
─────────────────────────────────────
Computer Use Agent — Tier 3 Differentiator.

Like Claude Computer Use / OpenAI Operator — but running locally.
Uses Gemini Vision to *see* the desktop, then pyautogui to *act* on it.

Flow:
  1. Take a screenshot
  2. Send to Gemini 2.0 Flash Vision with the goal
  3. Receive a structured JSON action
  4. Execute action (click/type/scroll/hotkey/open)
  5. Repeat until DONE or max_steps reached

Usage:
    agent = ComputerUseAgent(gemini_api_key="...")
    result = agent.execute("Open Notepad, type 'Hello World', and save the file")
"""

import os
import time
import base64
import json
import re
import logging
import threading
from typing import Optional, Tuple

logger = logging.getLogger("ComputerUseAgent")

try:
    import pyautogui
    pyautogui.FAILSAFE = True   # Move mouse to corner to abort
    _PYAUTOGUI = True
except ImportError:
    _PYAUTOGUI = False
    logger.warning("pyautogui not installed.")

try:
    from PIL import ImageGrab
    _PIL = True
except ImportError:
    _PIL = False

try:
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False


VISION_PROMPT = """You are an AI agent controlling a Windows desktop via mouse and keyboard.

Goal: {goal}

Steps completed so far:
{history}

Look at the screenshot and decide the SINGLE best next action.

Respond with ONLY a JSON object — no explanation:

  {{"action": "click",    "x": 500,    "y": 300,   "button": "left"}}
  {{"action": "type",     "text": "Hello World"}}
  {{"action": "hotkey",   "keys": ["ctrl", "s"]}}
  {{"action": "scroll",   "x": 500,    "y": 400,   "direction": "down", "clicks": 3}}
  {{"action": "press",    "key": "enter"}}
  {{"action": "open_app", "name": "notepad"}}
  {{"action": "wait",     "seconds": 1}}
  {{"action": "done",     "result": "Describe what was accomplished"}}
  {{"action": "need_help","question": "What you need clarified"}}

Important:
- Use actual pixel coordinates visible in the screenshot
- DONE only when the goal is FULLY complete
- If something failed, try an alternative approach"""


class ComputerUseAgent:
    """
    Vision-guided desktop automation agent.
    Sees the screen via Gemini Vision, acts via pyautogui.
    """

    def __init__(self, gemini_api_key: str = None, speak_fn=None):
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.speak = speak_fn or (lambda t, **kw: None)
        self._active = False
        self._lock = threading.Lock()

    # ─── Public API ───────────────────────────────────────────────────────────

    def execute(self, goal: str, max_steps: int = 20, speak_progress: bool = True) -> str:
        """Run an autonomous computer use session to achieve the goal."""
        with self._lock:
            if self._active:
                return "Computer Use Agent is already running."
            self._active = True

        try:
            return self._run(goal, max_steps, speak_progress)
        finally:
            self._active = False

    def stop(self):
        """Interrupt the running agent."""
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    # ─── Core loop ────────────────────────────────────────────────────────────

    def _run(self, goal: str, max_steps: int, speak_progress: bool) -> str:
        logger.info(f"[ComputerUse] Goal: {goal}")
        if speak_progress:
            self.speak(f"Starting computer use task: {goal[:40]}")

        history = []

        for step in range(max_steps):
            if not self._active:
                return f"Task interrupted after {step} steps."

            # 1. Capture screenshot
            screenshot_b64 = self._screenshot_b64()
            if not screenshot_b64:
                return "Failed to capture screen."

            if speak_progress and step > 0 and step % 3 == 0:
                self.speak(f"Step {step + 1} of {max_steps}.")

            # 2. Ask Gemini Vision for next action
            action = self._ask_vision(goal, history, screenshot_b64)
            if not action:
                logger.warning(f"[ComputerUse] Step {step+1}: No action returned")
                time.sleep(1)
                continue

            action_type = action.get("action", "").lower()
            logger.info(f"[ComputerUse] Step {step+1}: {action_type} → {action}")

            # 3. Handle terminal states
            if action_type == "done":
                result = action.get("result", "Task completed.")
                if speak_progress:
                    self.speak(f"Done. {result}")
                logger.info(f"[ComputerUse] DONE: {result}")
                return result

            if action_type == "need_help":
                question = action.get("question", "I need clarification.")
                if speak_progress:
                    self.speak(question)
                return f"[Needs input] {question}"

            # 4. Execute action
            result_str, success = self._execute_action(action)
            history.append({
                "step": step + 1,
                "action": action_type,
                "details": str(action)[:80],
                "result": result_str
            })

            if not success:
                logger.warning(f"[ComputerUse] Action failed: {result_str}")

            time.sleep(0.8)  # Let UI settle

        summary = f"Reached {max_steps} steps. Last: {history[-1]['action'] if history else 'none'}"
        if speak_progress:
            self.speak("Reached maximum steps for this computer use task.")
        return summary

    # ─── Action executor ──────────────────────────────────────────────────────

    def _execute_action(self, action: dict) -> Tuple[str, bool]:
        """Execute a single action and return (description, success)."""
        if not _PYAUTOGUI:
            return "pyautogui not available", False

        action_type = action.get("action", "").lower()
        try:
            if action_type == "click":
                x, y = int(action.get("x", 0)), int(action.get("y", 0))
                button = action.get("button", "left")
                pyautogui.moveTo(x, y, duration=0.3)
                pyautogui.click(button=button)
                return f"Clicked {button} at ({x},{y})", True

            elif action_type == "type":
                text = action.get("text", "")
                pyautogui.write(str(text), interval=0.04)
                return f"Typed: {text[:30]}", True

            elif action_type == "hotkey":
                keys = action.get("keys", [])
                if keys:
                    pyautogui.hotkey(*[str(k) for k in keys])
                return f"Hotkey: {'+'.join(str(k) for k in keys)}", True

            elif action_type == "scroll":
                x = int(action.get("x", pyautogui.size().width // 2))
                y = int(action.get("y", pyautogui.size().height // 2))
                clicks = int(action.get("clicks", 3))
                direction = action.get("direction", "down")
                pyautogui.moveTo(x, y, duration=0.2)
                pyautogui.scroll(clicks if direction == "up" else -clicks)
                return f"Scrolled {direction}", True

            elif action_type == "press":
                key = action.get("key", "enter")
                pyautogui.press(str(key))
                return f"Pressed: {key}", True

            elif action_type == "open_app":
                name = action.get("name", "")
                import subprocess
                subprocess.Popen(name, shell=True)
                time.sleep(1.5)  # Wait for app to open
                return f"Opened: {name}", True

            elif action_type == "wait":
                secs = min(float(action.get("seconds", 1)), 10)
                time.sleep(secs)
                return f"Waited {secs}s", True

            else:
                return f"Unknown action: {action_type}", False

        except Exception as e:
            logger.error(f"Action error ({action_type}): {e}")
            return f"Error: {e}", False

    # ─── Vision ───────────────────────────────────────────────────────────────

    def _screenshot_b64(self) -> Optional[str]:
        """Capture the full screen as base64 PNG, resized for API efficiency."""
        try:
            if _PIL:
                img = ImageGrab.grab()
                # Resize to 1280x720 for faster API calls
                img = img.resize((1280, 720))
                import io
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
            elif _PYAUTOGUI:
                img = pyautogui.screenshot()
                import io
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
        return None

    def _ask_vision(self, goal: str, history: list, screenshot_b64: str) -> Optional[dict]:
        """Ask Gemini 2.0 Flash Vision what action to take next."""
        if not self.api_key or not _HTTPX:
            logger.warning("No API key or httpx unavailable.")
            return None

        history_str = "\n".join(
            f"  Step {h['step']}: {h['action']} → {h['result']}"
            for h in history[-6:]
        ) or "  None yet."

        prompt = VISION_PROMPT.format(goal=goal, history=history_str)

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": screenshot_b64}}
                    ]
                }],
                "generationConfig": {"maxOutputTokens": 150, "temperature": 0.1}
            }
            with httpx.Client(timeout=25) as client:
                resp = client.post(url, json=payload)
            text = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

            # Extract JSON from the response
            match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.error(f"Vision API error: {e}")
        return None
