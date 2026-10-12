"""
sentinel/modules/vision.py
─────────────────────────
Multimodal Ambient Desktop Vision & Active Window Perception.
Combines real-time foreground window inspection with vision analysis.
"""

import base64
import logging
import os

import httpx

from sentinel.utils.window_context import get_active_window_info

logger = logging.getLogger("VisionModule")

try:
    from PIL import ImageGrab

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


class VisionModule:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.logger = logger

    def capture_screen(self, filename: str = "screenshot.png", crop_to_window: bool = False) -> str | None:
        """Captures the primary monitor screen or active window bounds."""
        if not _PIL_AVAILABLE:
            self.logger.warning("PIL not available for screen capture.")
            return None

        try:
            bbox = None
            if crop_to_window:
                info = get_active_window_info()
                rect = info.get("rect")
                if rect and rect[2] > rect[0] and rect[3] > rect[1]:
                    bbox = rect

            screenshot = ImageGrab.grab(bbox=bbox)
            screenshot.save(filename)
            return filename
        except Exception as e:
            self.logger.error(f"Failed to capture screen: {e}")
            return None

    def analyze_screen(
        self,
        prompt: str = "What is on my screen right now? Describe the user's active context in detail.",
        crop_to_window: bool = False,
    ) -> str:
        """
        Captures the screen and performs multimodal vision analysis.
        Degrades gracefully to window metadata if API key or cloud network is unavailable.
        """
        win_info = get_active_window_info()
        context_prefix = f"Active Window: '{win_info['title']}' (Application: {win_info['app_name']}). "

        img_path = self.capture_screen(crop_to_window=crop_to_window)

        # Fallback to local window context if image capture failed
        if not img_path:
            return f"{context_prefix} You are working in {win_info['app_name']} with document '{win_info['title']}'."

        try:
            with open(img_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            full_prompt = f"{context_prefix}\nUser Question: {prompt}"

            # 1. Try Gemini Cloud Vision if API key is present
            if self.api_key:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
                    payload = {
                        "contents": [
                            {
                                "parts": [
                                    {"text": full_prompt},
                                    {"inlineData": {"mimeType": "image/png", "data": base64_image}},
                                ]
                            }
                        ]
                    }
                    with httpx.Client(timeout=45) as client:
                        response = client.post(url, json=payload)
                        if response.status_code == 200:
                            result = response.json()
                            analysis = (
                                result.get("candidates", [{}])[0]
                                .get("content", {})
                                .get("parts", [{}])[0]
                                .get("text", "")
                            )
                            if analysis:
                                return analysis.strip()
                        else:
                            self.logger.warning(
                                f"Gemini Vision API returned {response.status_code}: {response.text[:100]}"
                            )
                except Exception as exc:
                    self.logger.warning(f"Cloud vision failed: {exc}. Trying local multimodal fallback.")

            # 2. Local Multimodal Fallback: Ollama (LLaVA / Moondream / Llama3.2-Vision)
            ollama_result = self._analyze_with_ollama(base64_image, full_prompt)
            if ollama_result:
                return ollama_result

            # 3. Graceful degradation: Window and process context
            return f"{context_prefix} Currently focusing on: {win_info['title']}."

        except Exception as e:
            self.logger.error(f"Vision analysis error: {e}")
            return f"{context_prefix} Analysis error: {e}"
        finally:
            if img_path and os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception:
                    pass

    def _analyze_with_ollama(self, base64_image: str, prompt: str) -> str | None:
        """
        Multimodal Local Vision Fallback via Ollama.
        Inspects locally installed models for vision capabilities (e.g. llava, moondream, llama3.2-vision).
        """
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        preferred_model = os.getenv("SENTINEL_OLLAMA_VISION_MODEL")

        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{ollama_host}/api/tags")
                if resp.status_code != 200:
                    return None
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]

            target_model = None
            if preferred_model and any(preferred_model in m for m in models):
                target_model = preferred_model
            else:
                for candidate in ["llava", "moondream", "llama3.2-vision", "bakllava", "minicpm"]:
                    matched = next((m for m in models if candidate in m.lower()), None)
                    if matched:
                        target_model = matched
                        break

            if not target_model:
                return None

            self.logger.info(f"Using local Ollama multimodal vision: model={target_model}")
            payload = {
                "model": target_model,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
            }
            with httpx.Client(timeout=60.0) as client:
                res = client.post(f"{ollama_host}/api/generate", json=payload)
                if res.status_code == 200:
                    out = res.json().get("response", "")
                    if out and out.strip():
                        return f"[Local Vision · {target_model}]\n" + out.strip()
        except Exception as e:
            self.logger.debug(f"Ollama local vision fallback not reachable: {e}")
        return None

    def describe_active_window(self) -> str:
        """Specifically describes the active foreground window context."""
        return self.analyze_screen(
            prompt="Analyze this active application window. What is the user working on? If code or terminal, identify file and errors. If browser or document, summarize the main content.",
            crop_to_window=True,
        )
