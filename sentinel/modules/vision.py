import os
import time
import base64
from PIL import ImageGrab
import httpx
import logging

class VisionModule:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.logger = logging.getLogger("VisionModule")

    def capture_screen(self, filename="screenshot.png"):
        """Captures the primary monitor screen."""
        try:
            screenshot = ImageGrab.grab()
            screenshot.save(filename)
            return filename
        except Exception as e:
            self.logger.error(f"Failed to capture screen: {e}")
            return None

    def analyze_screen(self, prompt="What is on my screen right now? Describe in detail."):
        """Captures the screen and sends it to Gemini 2.5 Flash for vision analysis."""
        img_path = self.capture_screen()
        if not img_path:
            return "Could not capture screen."

        try:
            with open(img_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": base64_image
                            }
                        }
                    ]
                }]
            }

            with httpx.Client(timeout=60) as client:
                response = client.post(url, json=payload)
                result = response.json()
                
            analysis = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return analysis or "No analysis received."
        except Exception as e:
            self.logger.error(f"Vision analysis error: {e}")
            return f"Vision analysis failed: {e}"
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)

    def describe_active_window(self):
        """Specifically tries to describe the context of the current active window."""
        return self.analyze_screen("Focus on the active window. What is the user doing? If it's a code editor, what language and task? If it's a browser, what is the website about?")
