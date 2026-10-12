"""
sentinel/commands/web_automation.py
───────────────────────────────────
Autonomous Web Browser Agent Command.
Handles interactive web navigation, scraping, and Playwright autonomous tasks.
"""

import logging
import webbrowser

logger = logging.getLogger("WebAutomation")


class WebAutomation:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    def execute(self, command: str, entity: str | None = None) -> str:
        text = (command or "").lower()
        if not text:
            return "What website or task would you like me to execute?"

        # 1. Autonomous Vision/Playwright tasks (e.g. search, buy, extract, scrape, check flights)
        is_autonomous_task = any(
            k in text
            for k in (
                "scrape",
                "extract",
                "find on amazon",
                "check flights",
                "research",
                "compare prices",
                "fill form on",
            )
        )

        if (
            is_autonomous_task
            and self.orchestrator
            and hasattr(self.orchestrator, "playwright_agent")
            and self.orchestrator.playwright_agent
        ):
            try:
                logger.info(f"Dispatching autonomous browser task: {command}")
                res = self.orchestrator.playwright_agent.perform_autonomous_task(command, max_steps=5)
                if res and "error" not in res.lower():
                    return f"Browser agent completed task: {res}"
            except Exception as e:
                logger.debug(f"Playwright autonomous agent error: {e}")

        # 2. Standard direct URL navigation
        if "." in text and ("http" in text or "www." in text or ".com" in text or ".org" in text):
            url = text.strip()
            if not url.startswith("http"):
                url = f"https://{url}"
            webbrowser.open(url)
            return f"Opening {url} in your default browser."

        for kw in ("open website", "navigate to", "browse to", "go to"):
            if kw in text:
                url = text.split(kw, 1)[1].strip()
                if not url.startswith("http"):
                    url = f"https://{url}"
                webbrowser.open(url)
                return f"Navigating to {url}."

        # Default web search fallback
        search_query = entity or command
        webbrowser.open(f"https://www.google.com/search?q={search_query}")
        return f"Searching the web for: '{search_query}'."
