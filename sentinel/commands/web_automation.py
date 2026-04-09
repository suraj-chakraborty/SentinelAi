import logging
import webbrowser

logger = logging.getLogger("WebAutomation")


class WebAutomation:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    def execute(self, command: str, entity: str) -> str:
        text = (command or "").lower()
        if not text:
            return "What website would you like me to open?"
        # Open URL if a URL-like string is present
        if "." in text and (text.startswith("http") or not text.startswith("http")):
            url = text if text.startswith("http") else f"https://{text}"
            webbrowser.open(url)
            return f"Opening {url} in your browser."
        if "open website" in text or "navigate to" in text or "browse to" in text:
            # extract the URL after the keyword
            for kw in ("open website", "navigate to", "browse to"):
                if kw in text:
                    url = text.split(kw,1)[1].strip()
                    if not url.startswith("http"):
                        url = "https://" + url
                    webbrowser.open(url)
                    return f"Opening {url} in your browser."
        return "Web automation command not recognised."
