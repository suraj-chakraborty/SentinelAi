"""
SentinelAI — Production Grade Modular AI Assistant
──────────────────────────────────────────────────
Main entry points and thin wrappers for the modular system.
"""

import sys
import os

import logging

# Ensure the project root is in the path for package imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Enable logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

from sentinel.app.boot import main as boot_main
from sentinel.app.intelligence import gemini_generate, interpret_command

if __name__ == "__main__":
    # Check for CLI flags before starting the main loop
    if "--install-deps" in sys.argv:
        import subprocess
        try:
            print("Installing Playwright browsers...")
            subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
            sys.exit(0)
        except Exception as e:
            print(f"Error installing dependencies: {e}")
            sys.exit(1)
            
    # Start the modular application
    boot_main()

# For legacy plugins or modules that still import from sentinel_ai
__all__ = ["gemini_generate", "interpret_command"]
