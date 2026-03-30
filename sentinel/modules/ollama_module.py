import ollama
import psutil
import logging
import subprocess

class OllamaModule:
    def __init__(self):
        self.logger = logging.getLogger("OllamaModule")
        self.model = self._choose_model()

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
            ollama.list()
            return True
        except Exception:
            return False

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
