import ollama
import psutil
import logging
import subprocess
import threading
import os

class OllamaModule:
    def __init__(self):
        self.logger = logging.getLogger("OllamaModule")
        self.model = self._choose_model()
        self.is_installing = False

    def _choose_model(self):
        """Chooses an appropriate model based on system specs."""
        try:
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            # Check for GPU (simplified)
            has_gpu = False
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    has_gpu = True
            except:
                pass

            if ram_gb > 16 or has_gpu:
                self.logger.info(f"High-spec detected ({ram_gb:.1f}GB RAM, GPU: {has_gpu}). Choosing Llama3.")
                return "llama3"
            else:
                self.logger.info(f"Low-spec detected ({ram_gb:.1f}GB RAM). Choosing TinyLlama.")
                return "tinyllama"
        except Exception as e:
            self.logger.error(f"Error detecting specs: {e}")
            return "tinyllama"

    def is_available(self):
        """Checks if the Ollama service is running."""
        try:
            # Using list() as a cheap ping
            import ollama
            ollama.list()
            return True
        except Exception:
            return False

    def start_service(self):
        """Attempts to start the Ollama service in the background."""
        if self.is_available():
            return True
        
        self.logger.info("Service Offline. Attempting to start Ollama...")
        try:
            import platform
            if platform.system() == "Windows":
                # Try starting the background server
                res = subprocess.run(["where", "ollama"], capture_output=True, text=True)
                if res.returncode != 0:
                    # Not in PATH, need to install
                    self.install_service()
                    return False

                subprocess.Popen(["ollama", "serve"], 
                                creationflags=subprocess.CREATE_NO_WINDOW,
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(["ollama", "serve"], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            self.logger.error(f"Failed to start Ollama service: {e}")
            self.install_service()
            return False

    def install_service(self):
        """Automatically downloads and launches the Ollama installer for Windows."""
        if self.is_installing:
            return
            
        import requests
        from pathlib import Path
        
        # Sentinel AppData folder for the installer
        target_dir = Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "SentinelAi"
        target_dir.mkdir(parents=True, exist_ok=True)
        installer_path = target_dir / "OllamaSetup.exe"
        
        url = "https://ollama.com/download/OllamaSetup.exe"
        
        def _download_task():
            self.is_installing = True
            try:
                self.logger.info(f"Downloading Ollama installer from {url}...")
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(installer_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                self.logger.info("Download complete. Launching installer...")
                # Launch the installer (this will pop up a window for the user)
                subprocess.Popen([str(installer_path)])
                
            except Exception as e:
                self.logger.error(f"Ollama installation failed: {e}")
                self.is_installing = False
        
        threading.Thread(target=_download_task, daemon=True).start()

    def generate(self, prompt):
        """Generates a response using the local model."""
        if not self.is_available():
            return "Ollama service is not running. Please start it for offline support."

        try:
            # Ensure model is pulled
            self.logger.info(f"Generating offline with {self.model}...")
            response = ollama.generate(model=self.model, prompt=prompt)
            return response['response']
        except Exception as e:
            if "not found" in str(e).lower():
                return f"Model {self.model} not found. Please run 'ollama pull {self.model}' in terminal."
            self.logger.error(f"Ollama generation error: {e}")
            return f"Offline generation failed: {e}"

    def pull_model_if_missing(self):
        """Proactively pulls the selected model if it's not in the list."""
        try:
            models = [m['name'] for m in ollama.list()['models']]
            if self.model not in [m.split(':')[0] for m in models]:
                self.logger.info(f"Pulling {self.model} model...")
                ollama.pull(self.model)
        except Exception as e:
            self.logger.error(f"Could not pull model: {e}")
