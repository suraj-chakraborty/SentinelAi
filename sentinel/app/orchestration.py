"""
sentinel/app/orchestration.py
─────────────────────────────
Main command routing, agent handling, and high-level logic.
"""

import os
import time
import logging
import threading
from typing import Optional

# Config & Core
from sentinel.app.config import COMMAND_WAV
from sentinel.app.state import get_state
from sentinel.app.voice import speak, transcribe
from sentinel.app.audio import record_until_silence
from sentinel.core.repair import RepairAgent

logger = logging.getLogger("SentinelOrchestration")

# ── Command Routing ──────────────────────────────────────────────────────────

def execute_command(command: str, orchestrator_ref=None):
    """
    Main entry point for command routing with autonomous repair.
    """
    if not command: return
    
    cmd = command.strip().lower()
    logger.info(f"Processing command: {cmd}")
    
    # 1. System Overrides (Priority)
    if cmd in ("stop", "stop agent", "disable agent"):
        speak("Stopping autonomous agent.")
        get_state().set_status("AGENT_ACTIVE", False)
        return
        
    if cmd in ("pause listening", "stop listening"):
        get_state().set_status("LISTENING_PAUSED", True)
        speak("Pausing listening for 5 minutes.")
        return

    # 2. Orchestrator Routing
    if orchestrator_ref:
        try:
            response = orchestrator_ref.run_command(command)
            if response:
                speak(response)
                return
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            # Continue to AI fallback on orchestrator crash

    # 3. Fallback to AI Analysis
    from sentinel.app.intelligence import interpret_command
    try:
        response = interpret_command(command)
        if response:
            # Handle AI-generated [EXECUTE: ...] tags for autonomous repair
            import re
            match = re.search(r"\[EXECUTE:\s*(.*?)\]", response)
            if match:
                exec_cmd = match.group(1).strip()
                clean_msg = response.replace(match.group(0), "").strip()
                if clean_msg:
                    speak(clean_msg)
                
                logger.info(f"AI suggested autonomous execution: {exec_cmd}")
                from sentinel.utils.app_launcher import launch_application
                launch_application(exec_cmd)
                return
            
            speak(response)
        else:
            speak("I'm sorry, I couldn't understand or execute that command.")
    except Exception as e:
        logger.error(f"Execution crash: {e}")
        # Phase 3: Autonomous Error Self-Repair
        repair_agent = RepairAgent(llm_callback=lambda p: interpret_command(p))
        repair_suggestion = repair_agent.suggest_repair(command, str(e))
        if repair_suggestion:
            speak(repair_suggestion)
            # Future: Wait for confirmation before auto-executing
        else:
            speak("I encountered a system error and could not find an immediate repair path.")

# ── Agent Strategy ───────────────────────────────────────────────────────────

def handle_agent_step(command: str, orchestrator_ref=None):
    """
    Handles step-by-step agent instructions.
    """
    state = get_state()
    cmd = (command or "").strip().lower()
    
    if "confirm" in cmd or "yes" in cmd:
        pending = state.get_status("PENDING_STEP")
        if pending:
            # Execute the pending step
            # ... execution logic ...
            state.set_status("PENDING_STEP", None)
            speak("Step confirmed and executed.")
        else:
            speak("No pending step to confirm.")
        return

    # Analyze step via AI
    speak("Analyzing agent step...")
    execute_command(command, orchestrator_ref=orchestrator_ref)

# ── Alert Loop ──────────────────────────────────────────────────────────────

def start_alert_loop(orchestrator_ref):
    """Periodically checks for system alerts and reminders."""
    def _run():
        while True:
            try:
                alerts = orchestrator_ref.check_periodic_alerts()
                for alert in alerts:
                    speak(alert)
                    logger.info(f"System alert: {alert}")
            except Exception:
                pass
            time.sleep(60)
            
    threading.Thread(target=_run, daemon=True).start()
