"""
sentinel/modules/playwright_agent.py
──────────────────────────────────────
Full vision-guided Playwright agent — Tier 2 Upgrade.

Upgraded from a skeletal Google search stub to a real LLM-driven
browser automation agent that:
  1. Navigates to a URL / search
  2. Screenshots the page
  3. Sends the screenshot to Gemini Vision for analysis
  4. Receives structured actions (click selector, type text, scroll, etc.)
  5. Executes them and loops until the goal is achieved or max_steps reached
"""

import os
import base64
import time
import logging
import io
from typing import Optional

logger = logging.getLogger("PlaywrightAgentModule")

try:
    from playwright.sync_api import sync_playwright, Page, Browser
    _PW_AVAILABLE = True
except ImportError:
    _PW_AVAILABLE = False
    logger.warning("playwright not available. Run: playwright install chromium")

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False


AGENT_SYSTEM_PROMPT = """You are a browser automation agent. You are given a screenshot of the current browser page and a goal.

Respond with a JSON action:
  {{"action": "click", "selector": "CSS selector or text"}}
  {{"action": "type", "selector": "CSS selector", "text": "text to type"}}
  {{"action": "navigate", "url": "https://..."}}
  {{"action": "scroll", "direction": "down"}}
  {{"action": "press", "key": "Enter"}}
  {{"action": "done", "result": "What was accomplished"}}
  {{"action": "fail", "reason": "Why the task cannot be completed"}}

Current goal: {goal}
Steps taken so far: {steps}

Analyze the screenshot and return ONLY the JSON action. No explanation."""


class PlaywrightAgentModule:
    """Vision-guided browser automation agent using Playwright + Gemini Vision."""

    def __init__(self, headless: bool = False, gemini_api_key: str = None):
        self.headless = headless
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.logger = logging.getLogger("PlaywrightAgentModule")
        self.browser: Optional["Browser"] = None
        self.page: Optional["Page"] = None
        self._pw = None

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> tuple:
        """Initialize Playwright with Chromium."""
        if not _PW_AVAILABLE:
            return False, "Playwright not installed. Run: pip install playwright && playwright install chromium"
        try:
            self._pw = sync_playwright().start()
            self.browser = self._pw.chromium.launch(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"]
            )
            context = self.browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            self.page = context.new_page()
            self.logger.info("Playwright started.")
            return True, "Browser started."
        except Exception as e:
            self.logger.error(f"Playwright start error: {e}")
            return False, f"Error: {e}"

    def stop(self):
        """Close browser and clean up."""
        try:
            if self.browser:
                self.browser.close()
            if self._pw:
                self._pw.stop()
            self.browser = None
            self.page = None
            self.logger.info("Playwright stopped.")
        except Exception:
            pass

    def navigate_to(self, url: str) -> tuple:
        """Navigate to a URL, starting browser if needed."""
        if not self.page:
            ok, msg = self.start()
            if not ok:
                return False, msg
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            return True, f"Navigated to {url}"
        except Exception as e:
            self.logger.error(f"Navigation error: {e}")
            return False, f"Error: {e}"

    # ─── Vision-Guided Autonomous Loop ────────────────────────────────────────

    def perform_autonomous_task(self, task_description: str, max_steps: int = 10) -> str:
        """
        Full LLM-driven browser automation loop.
        Takes a screenshot after each action and asks Gemini what to do next.
        """
        if not self.page:
            ok, msg = self.start()
            if not ok:
                return f"Could not start browser: {msg}"

        steps_taken = []
        self.logger.info(f"[PlaywrightAgent] Goal: {task_description}")

        # Start with a Google search if no URL specified
        if not task_description.startswith("http"):
            search_url = f"https://www.google.com/search?q={task_description.replace(' ', '+')}"
            try:
                self.page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e:
                return f"Failed to start search: {e}"

        for step in range(max_steps):
            self.logger.info(f"[PlaywrightAgent] Step {step + 1}/{max_steps}")

            # 1. Screenshot current state
            screenshot_b64 = self._capture_screenshot_b64()
            if not screenshot_b64:
                break

            # 2. Ask Gemini Vision what to do next
            action = self._get_next_action(task_description, steps_taken, screenshot_b64)
            if not action:
                self.logger.warning("No action returned from vision model.")
                break

            action_type = action.get("action", "").lower()
            self.logger.info(f"[PlaywrightAgent] Action: {action}")

            # 3. Execute action
            if action_type == "done":
                result = action.get("result", "Task completed.")
                self.logger.info(f"[PlaywrightAgent] DONE: {result}")
                return result

            elif action_type == "fail":
                reason = action.get("reason", "Unknown failure.")
                self.logger.warning(f"[PlaywrightAgent] FAIL: {reason}")
                return f"Could not complete task: {reason}"

            elif action_type == "navigate":
                url = action.get("url", "")
                try:
                    self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    steps_taken.append(f"Navigated to {url}")
                except Exception as e:
                    steps_taken.append(f"Navigate failed: {e}")

            elif action_type == "click":
                selector = action.get("selector", "")
                try:
                    self.page.click(selector, timeout=5000)
                    steps_taken.append(f"Clicked: {selector}")
                except Exception:
                    # Try clicking by visible text
                    try:
                        self.page.get_by_text(selector).first.click(timeout=3000)
                        steps_taken.append(f"Clicked text: {selector}")
                    except Exception as e2:
                        steps_taken.append(f"Click failed: {e2}")

            elif action_type == "type":
                selector = action.get("selector", "")
                text = action.get("text", "")
                try:
                    self.page.fill(selector, text, timeout=5000)
                    steps_taken.append(f"Typed '{text[:20]}' into {selector}")
                except Exception as e:
                    steps_taken.append(f"Type failed: {e}")

            elif action_type == "scroll":
                direction = action.get("direction", "down")
                amount = 500 if direction == "down" else -500
                try:
                    self.page.evaluate(f"window.scrollBy(0, {amount})")
                    steps_taken.append(f"Scrolled {direction}")
                except Exception as e:
                    steps_taken.append(f"Scroll failed: {e}")

            elif action_type == "press":
                key = action.get("key", "Enter")
                try:
                    self.page.keyboard.press(key)
                    steps_taken.append(f"Pressed: {key}")
                except Exception as e:
                    steps_taken.append(f"Press failed: {e}")

            else:
                self.logger.warning(f"Unknown action type: {action_type}")
                steps_taken.append(f"Unknown action: {action_type}")

            time.sleep(1.5)  # Let the page settle after each action

        return f"Autonomous task ended after {len(steps_taken)} steps. Progress: {'; '.join(steps_taken[-3:])}"

    # ─── Simple helpers ───────────────────────────────────────────────────────

    def search_and_get_text(self, query: str) -> str:
        """Quick helper: search Google, return first result page text."""
        if not self.page:
            self.start()
        try:
            self.page.goto(f"https://www.google.com/search?q={query.replace(' ', '+')}", timeout=20000)
            time.sleep(2)
            # Extract visible text
            text = self.page.inner_text("body")
            return text[:3000] if text else "No content extracted."
        except Exception as e:
            return f"Search failed: {e}"

    # ─── Private ──────────────────────────────────────────────────────────────

    def _capture_screenshot_b64(self) -> Optional[str]:
        """Capture page screenshot and return as base64 PNG."""
        try:
            if not self.page:
                return None
            png_bytes = self.page.screenshot(full_page=False)
            return base64.b64encode(png_bytes).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Screenshot failed: {e}")
            return None

    def _get_next_action(self, goal: str, steps: list, screenshot_b64: str) -> Optional[dict]:
        """Ask Gemini Vision to decide the next browser action."""
        import json, re
        if not self.gemini_api_key or not _HTTPX_AVAILABLE:
            return None
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={self.gemini_api_key}"
            prompt = AGENT_SYSTEM_PROMPT.format(
                goal=goal,
                steps="; ".join(steps[-5:]) if steps else "None yet"
            )
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": screenshot_b64}}
                    ]
                }],
                "generationConfig": {"maxOutputTokens": 200, "temperature": 0.2}
            }
            with httpx.Client(timeout=30) as client:
                resp = client.post(url, json=payload)
                j = resp.json()

            text = j.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return None
        except Exception as e:
            self.logger.error(f"Vision LLM error: {e}")
            return None
