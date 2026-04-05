"""
sentinel/modules/ollama_module.py
──────────────────────────────────
Local LLM integration via the Ollama daemon.

Improvements over v1:
  - Lazy `import ollama` — won't crash if package is absent at startup
  - Cached `is_available()` result (30-second TTL) to avoid repeated pings
  - `pull_model_if_missing()` called automatically during orchestrator init
  - `generate_streaming()` for low-latency progressive token output
  - Proper model selection: llama3 for high-spec, tinyllama for low-spec
  - Auto-installs Ollama if the binary is missing (Windows only)
"""

import logging
import os
import subprocess
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("OllamaModule")


class OllamaModule:
    """Manages the local Ollama daemon and LLM inference."""

    _INSTALL_URL = "https://ollama.com/download/OllamaSetup.exe"
    _AVAILABILITY_TTL = 30          # seconds between ping checks
    _HIGH_SPEC_RAM_GB = 16

    def __init__(self):
        self.model: str = self._choose_model()
        self.is_installing: bool = False
        self._available_cache: Optional[bool] = None
        self._last_check: float = 0.0
        self._lock = threading.Lock()

    # ── Model selection ───────────────────────────────────────────────────────

    def _choose_model(self) -> str:
        """Pick llama3 for high-spec machines, tinyllama otherwise."""
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            has_gpu = self._detect_gpu()
            if ram_gb >= self._HIGH_SPEC_RAM_GB or has_gpu:
                logger.info("High-spec detected (%.1f GB RAM, GPU=%s). Using llama3.", ram_gb, has_gpu)
                return "llama3"
            logger.info("Low-spec detected (%.1f GB RAM). Using tinyllama.", ram_gb)
            return "tinyllama"
        except Exception as exc:
            logger.warning("Spec detection failed: %s. Defaulting to tinyllama.", exc)
            return "tinyllama"

    @staticmethod
    def _detect_gpu() -> bool:
        try:
            import GPUtil
            return bool(GPUtil.getGPUs())
        except Exception:
            return False

    # ── Availability check ────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """
        Check if the Ollama daemon is reachable.
        Result is cached for `_AVAILABILITY_TTL` seconds so callers can
        call this frequently without hammering the service.
        """
        now = time.monotonic()
        with self._lock:
            if self._available_cache is not None and (now - self._last_check) < self._AVAILABILITY_TTL:
                return self._available_cache
            result = self._ping()
            self._available_cache = result
            self._last_check = now
            return result

    def _ping(self) -> bool:
        try:
            import ollama as _ollama
            _ollama.list()
            return True
        except ImportError:
            logger.debug("ollama Python package not installed.")
            return False
        except Exception:
            return False

    def invalidate_cache(self):
        """Force next `is_available()` call to re-ping the service."""
        with self._lock:
            self._available_cache = None

    # ── Service lifecycle ─────────────────────────────────────────────────────

    def start_service(self) -> bool:
        """Attempt to start the Ollama background daemon."""
        if self.is_available():
            return True
        logger.info("Ollama offline — attempting to start daemon…")
        try:
            import platform
            if platform.system() == "Windows":
                result = subprocess.run(["where", "ollama"], capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning("Ollama binary not found in PATH — triggering install.")
                    self.install_service()
                    return False
                subprocess.Popen(
                    ["ollama", "serve"],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            self.invalidate_cache()
            return True
        except Exception as exc:
            logger.error("Failed to start Ollama: %s", exc)
            self.install_service()
            return False

    def install_service(self):
        """Download and launch the Ollama installer (Windows only)."""
        if self.is_installing:
            return

        appdata = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi")
        os.makedirs(appdata, exist_ok=True)
        installer_path = os.path.join(appdata, "OllamaSetup.exe")

        def _download():
            self.is_installing = True
            try:
                import requests
                logger.info("Downloading Ollama installer from %s …", self._INSTALL_URL)
                with requests.get(self._INSTALL_URL, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(installer_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                logger.info("Download complete. Launching installer…")
                subprocess.Popen([installer_path])
            except Exception as exc:
                logger.error("Ollama installation failed: %s", exc)
            finally:
                self.is_installing = False

        threading.Thread(target=_download, daemon=True).start()

    # ── Model management ──────────────────────────────────────────────────────

    def pull_model_if_missing(self):
        """Pull the selected model if it is not locally available. Safe to call in a thread."""
        if not self.is_available():
            return
        try:
            import ollama as _ollama
            models_resp = _ollama.list()
            local_names = []
            for m in (models_resp.get("models") or []):
                name = m.get("name") or m.get("model")
                if name:
                    local_names.append(name.split(":")[0])
            if self.model not in local_names:
                logger.info("Pulling model '%s' — this may take a few minutes…", self.model)
                _ollama.pull(self.model)
                logger.info("Model '%s' ready.", self.model)
        except Exception as exc:
            logger.error("Could not pull model '%s': %s", self.model, exc)

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """Blocking text generation. Returns the full response string."""
        if not self.is_available():
            return "Ollama service is not running. Please start it for offline support."
        try:
            import ollama as _ollama
            logger.debug("Generating offline with %s …", self.model)
            response = _ollama.generate(model=self.model, prompt=prompt)
            return response.get("response", "")
        except ImportError:
            return "Ollama Python package is not installed (pip install ollama)."
        except Exception as exc:
            if "not found" in str(exc).lower():
                return (
                    f"Model '{self.model}' not found locally. "
                    f"Run: ollama pull {self.model}"
                )
            logger.error("Ollama generation error: %s", exc)
            return f"Offline generation failed: {exc}"

    def generate_streaming(self, prompt: str, callback: Callable[[str], None]) -> str:
        """
        Stream tokens to `callback(token_str)` as they are generated.
        Returns the full assembled response when complete.
        """
        if not self.is_available():
            msg = "Ollama service is not running."
            callback(msg)
            return msg
        try:
            import ollama as _ollama
            full_response = []
            for chunk in _ollama.generate(model=self.model, prompt=prompt, stream=True):
                token = chunk.get("response", "")
                if token:
                    full_response.append(token)
                    callback(token)
            return "".join(full_response)
        except ImportError:
            msg = "Ollama Python package is not installed."
            callback(msg)
            return msg
        except Exception as exc:
            logger.error("Ollama streaming error: %s", exc)
            msg = f"Streaming failed: {exc}"
            callback(msg)
            return msg
