"""
sentinel/app/boot.py
────────────────────
Application entry point, pre-flight checks, and first-run wizard.
"""

import os
import sys
import subprocess
import threading
import logging
from typing import Optional

# Config & Registry
from sentinel.app.config import (
    ENV_PATH, ensure_appdata_dir, load_settings,
    save_settings, APPDATA_DIR, get_porcupine_model_path
)
from sentinel.app.registry import build_app_registry
from sentinel.app.state import get_state

# Core Orchestration
from sentinel.core.orchestrator import SentinelOrchestrator
from sentinel.app.voice import speak, transcribe
from sentinel.app.secrets import check_master_exists, verify_master
from sentinel.app.monitor import HealthMonitor

logger = logging.getLogger("SentinelBoot")

# ── Noise Suppression ────────────────────────────────────────────────────────

def suppress_noise():
    """Silence noisy third-party logging to keep the console clean."""
    logging.getLogger("pyaudio").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.ERROR)
    
    # Silence win32com's GenPy noisy outputs
    if os.name == "nt":
        # sys.stderr = open(os.devnull, 'w') # Disabled to allow debugging of startup errors
        pass

# ── Pre-flight checks ───────────────────────────────────────────────────────

def preflight_bootstrap():
    """Prepare folders and register apps on startup."""
    ensure_appdata_dir()
    settings = load_settings()
    
    # 1. Autostart check
    if settings.get("autostart", True):
        _ensure_autostart_reg()
        
    # 2. Rebuild app registry in background
    threading.Thread(target=build_app_registry, kwargs={"fast_only": False}, daemon=True).start()

def _ensure_autostart_reg():
    """Set Windows registry or task for autostart."""
    if os.name != "nt": return
    import winreg
    try:
        exe_path = sys.executable if getattr(sys, 'frozen', False) else None
        if exe_path:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run", 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, "SentinelAI", 0, winreg.REG_SZ, f'"{exe_path}"')
    except Exception:
        pass

# ── Orchestrator Setup ──────────────────────────────────────────────────────

def initialize_orchestrator() -> Optional[SentinelOrchestrator]:
    """Factory for the main Sentinel intelligence engine."""
    from sentinel.app.intelligence import gemini_generate
    
    try:
        orch = SentinelOrchestrator(
            llm_callback=lambda p: gemini_generate(p),
            speak_fn=speak,
            exit_callback=lambda: os._exit(0)
        )
        return orch
    except Exception as e:
        logger.error(f"Orchestrator init failed: {e}")
        return None

# ── Main Entry ──────────────────────────────────────────────────────────────

def main():
    """Sentinel AI main entry loop."""
    suppress_noise()
    preflight_bootstrap()
    
    # 1. Check Secrets Vault
    if not check_master_exists():
        logger.info("First run: Master password required.")
        # Trigger Wizard UI (Placeholder)
        print(">>> INITIALIZING SENTINEL VAULT...")
        
    # 2. Init Application & GUI
    from sentinel.app.application import SentinelApp
    
    # 3. Start Health Monitor
    monitor = HealthMonitor(interval=60)
    monitor.start()
    
    app = SentinelApp()
    
    logger.info("Sentinel Boot Sequence Complete. Starting Application...")
    app.run()
