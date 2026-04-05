import logging
import os
import subprocess

logger = logging.getLogger("OpenApp")

class OpenApp:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def execute(self, command: str, app_name: str) -> str:
        """
        Logic for searching and opening an application.
        """
        if not app_name:
            return "What application would you like me to open?"

        # 1. Handle common websites/tools
        websites = {
            "google": "https://www.google.com",
            "youtube": "https://www.youtube.com",
            "gmail": "https://mail.google.com",
            "gmail app": "https://mail.google.com",
            "facebook": "https://www.facebook.com",
            "github": "https://www.github.com",
            "twitter": "https://www.twitter.com",
            "reddit": "https://www.reddit.com",
            "browser": "https://www.google.com",
            "internet": "https://www.google.com",
            "bing": "https://www.bing.com",
        }
        
        # Clean target: lowercase, strip spaces and punctuation
        target = app_name.lower().strip().strip(".!?,")
        
        if target in websites:
            import webbrowser
            url = websites[target]
            logger.info(f"Launching browser for website: {target} -> {url}")
            success = webbrowser.open(url)
            if success:
                return f"Opening {target} in your browser..."
            else:
                logger.warning(f"webbrowser.open failed to return success for {url}")
                return f"I tried to open {target}, but your browser didn't respond."

        # New: Direct URL/Browse handling
        import webbrowser
        # Check if target looks like a URL (e.g. google.com, surajchakraborty.netlify.app)
        if "." in target and len(target.split(".")[-1]) >= 2:
            url = target if target.startswith("http") else f"https://{target}"
            logger.info(f"Direct URL navigation: {url}")
            if webbrowser.open(url):
                return f"Navigating to {target}..."

        # 2. Local Windows launch (registry index via sentinel_ai when available)
        try:
            from sentinel.utils.app_launcher import launch_application

            if launch_application(target):
                return f"Opening {target}..."
            return f"Sorry, I couldn't find the application {target}."
        except Exception as e:
            logger.error(f"Error in OpenApp plugin: {e}")
            return f"Failed to open {app_name}: {e}"
