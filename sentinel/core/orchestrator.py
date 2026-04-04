import os
import logging
import json
import time
import threading

from sentinel.modules.system_monitor import SystemMonitor
from sentinel.modules.coding import CodingModule
from sentinel.modules.calendar import CalendarModule
from sentinel.modules.web_agent import WebAgentModule
from sentinel.modules.vision import VisionModule
from sentinel.modules.knowledge_base import KnowledgeBaseModule
from sentinel.modules.iot_hub import IoTHubModule
from sentinel.modules.emotion import EmotionModule
from sentinel.modules.playwright_agent import PlaywrightAgentModule
from sentinel.modules.ollama_module import OllamaModule
from sentinel.modules.automation_module import AutomationModule
from sentinel.modules.file_intelligence_module import FileIntelligenceModule
from sentinel.modules.gesture_module import GestureModule
from sentinel.modules.habit_learner import HabitLearnerModule
from sentinel.modules.security_shield import SecurityShieldModule
from sentinel.modules.daily_brief import DailyBriefModule
from sentinel.modules.lora_collector import LoRACollector
from sentinel.core.agentic_engine import AgenticEngine
from sentinel.core.swarm_engine import SwarmOrchestrator
from sentinel.core.perception_stream import PerceptionStream
from sentinel.utils.notifier import WindowsNotifier
from sentinel.core.web_server import SentinelWebServer
from sentinel.core.command_router import CommandRouter

# Sci-Fi Tier Modules
from sentinel.modules.robotics_module import RoboticsModule
from sentinel.modules.bci_module import BCIModule
from sentinel.security.proxy_shield import ProxyShield
from sentinel.modules.quantum_module import QuantumModule

# Tier 3 new modules
try:
    from sentinel.core.computer_use_agent import ComputerUseAgent
    _COMPUTER_USE = True
except ImportError:
    _COMPUTER_USE = False

try:
    from sentinel.security.sandbox import Sandbox
    _SANDBOX = True
except ImportError:
    _SANDBOX = False

class SentinelOrchestrator:
    def __init__(self, llm_callback, speak_fn=None, exit_callback=None, master_key=None):
        self.llm_callback = llm_callback
        self.speak_fn = speak_fn
        self.exit_callback = exit_callback
        self.logger = logging.getLogger("SentinelOrchestrator")

        # ── Core Modules ──────────────────────────────────────────────────
        self.system_monitor    = self._init_module("SystemMonitor", SystemMonitor)
        self.coding_module     = self._init_module("CodingModule", CodingModule)
        self.calendar_module   = self._init_module("CalendarModule", CalendarModule)
        self.web_agent_module  = self._init_module("WebAgentModule", WebAgentModule)
        self.vision_module     = self._init_module("VisionModule", VisionModule)
        self.knowledge_base    = self._init_module(
            "KnowledgeBaseModule", KnowledgeBaseModule, master_key=master_key, llm_callback=self.llm_callback
        )
        self.iot_hub           = self._init_module("IoTHubModule", IoTHubModule)
        self.emotion_module    = self._init_module("EmotionModule", EmotionModule)
        self.playwright_agent  = self._init_module("PlaywrightAgentModule", PlaywrightAgentModule)
        self.ollama_module     = self._init_module("OllamaModule", OllamaModule)
        self.automation_module = self._init_module("AutomationModule", AutomationModule)
        self.gesture_module    = self._init_module("GestureModule", GestureModule)
        self.notifier          = self._init_module("WindowsNotifier", WindowsNotifier)
        self.file_intel        = self._init_module("FileIntelligenceModule", FileIntelligenceModule, self.knowledge_base) if self.knowledge_base else None
        self.habit_learner     = self._init_module("HabitLearnerModule", HabitLearnerModule, self.knowledge_base) if self.knowledge_base else None
        self.security_shield   = self._init_module("SecurityShieldModule", SecurityShieldModule, self.notifier, gemini_fn=self.llm_callback) if self.notifier else None
        self.daily_brief       = self._init_module("DailyBriefModule", DailyBriefModule, self)
        
        # ── Intelligence Engines ──────────────────────────────────────────
        self.agentic_engine    = self._init_module("AgenticEngine", AgenticEngine, self, speak_fn=self.speak_fn, gemini_fn=self.llm_callback)
        self.swarm_engine      = None
        self.perception_stream = None
        self.lora_collector    = None
        self.web_server        = self._init_module("SentinelWebServer", SentinelWebServer, self)

        # ── Tier 3 Modules ────────────────────────────────────────────────
        self.computer_use = None
        if _COMPUTER_USE:
            self.computer_use = self._init_module(
                "ComputerUseAgent",
                ComputerUseAgent,
                gemini_api_key=None,
                speak_fn=self.speak_fn,
            )
        # Manifest plugins load inside CommandRouter (single load path).

        self.sandbox = None
        if _SANDBOX:
            self.sandbox = Sandbox

        # ── Sci-Fi Modules ────────────────────────────────────────────────
        self.robotics = self._init_module("RoboticsModule", RoboticsModule)
        self.bci = self._init_module("BCIModule", BCIModule, self)
        self.quantum = self._init_module("QuantumModule", QuantumModule)
        self.proxy_shield = None
        
        # ── Modular Command Engine ───────────────────────────────────────
        self.command_router = CommandRouter(self)
        
        # Start background services
        try:
            if self.web_server:
                self.web_server.start()
            if self.habit_learner:
                self.habit_learner.start_monitoring()
            if self.security_shield:
                self.security_shield.start_shield()
            # Note: PerceptionStream must be manually enabled by the user for privacy
            self._start_proactive_loop()
        except Exception as e:
            self.logger.error(f"Failed to start background services: {e}")

    def _init_module(self, name, ctor, *args, **kwargs):
        try:
            return ctor(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"{name} init failed: {e}")
            return None

    def _ensure_optional_module(self, attr_name, module_name, ctor, *args, **kwargs):
        current = getattr(self, attr_name)
        if current is None:
            current = self._init_module(module_name, ctor, *args, **kwargs)
            setattr(self, attr_name, current)
        return current

    def set_master_key(self, key):
        """Updates the master key for encrypted modules."""
        if self.knowledge_base:
            self.knowledge_base.master_key = key
            self.logger.info("Master key updated for Knowledge Base.")

    def _start_proactive_loop(self):
        """Starts Phase 2: Autonomous Cron & Proactive Intelligence Loop."""
        if not self.vision_module or not self.notifier:
            return
        def _loop():
            self.logger.info("Autonomous Cron loop started (20m interval).")
            while True:
                try:
                    # Sleep for 20 minutes to preserve free tier Gemini quota
                    time.sleep(1200) 
                    
                    self.logger.info("Running proactive context analysis...")
                    
                    # Merge static screen analysis with multi-modal perception stream
                    context = ""
                    if self.perception_stream:
                        try:
                            context = self.perception_stream.get_recent_context(minutes=15)
                        except Exception:
                            pass

                    analysis = self.vision_module.analyze_screen(
                        f"Recent Context: {context}\nAnalyze the user's current activity. If you see a way to help (e.g., debugging a visible error, summarizing a long page, or automating a repetitive task), provide a short, proactive suggestion. If everything looks normal, return 'NORMAL'."
                    )
                    
                    # If we got a real suggestion (not 'NORMAL' and not an error message)
                    if analysis and "NORMAL" not in analysis.upper() and len(analysis) > 10:
                        # Check if it looks like a system error from our unified LLM wrapper
                        if "trouble connecting" not in analysis and "API key" not in analysis:
                            self.notifier.show_notification("Sentinel Proactive Alert", analysis)
                            self.logger.info(f"Proactive suggestion sent: {analysis}")
                        else:
                            self.logger.warning("Skipping proactive notification due to LLM connectivity issues.")
                except Exception as e:
                    self.logger.error(f"Proactive loop error: {e}")
        
        threading.Thread(target=_loop, daemon=True).start()

    def _safe_llm_call(self, prompt):
        """Calls the online LLM and falls back to Ollama if it fails or if offline."""
        try:
            return self.llm_callback(prompt)
        except Exception as e:
            self.logger.warning(f"Online LLM failed, falling back to Ollama: {e}")
            if self.ollama_module and self.ollama_module.is_available():
                return self.ollama_module.generate(prompt)
            return "Both online and offline LLM services are currently unavailable."

    def run_command(self, command):
        """Dispatches a command through the modular NLP-powered command engine."""
        if not command:
            return None
            
        self.logger.info(f"Orchestrating command: {command}")
        
        # Use the new modular router
        result = self.command_router.run_command(command)
        
        if result:
            self._record_lora(command, result)
            return result
            
        return "I'm sorry, I couldn't understand or execute that command."

    def _is_complex_command(self, command):
        """Heuristic to detect if a command needs decomposition (contains 'and', multiple verbs, etc.)."""
        keywords = [" and ", " then ", " then do ", " then also ", " after that "]
        return any(k in command for k in keywords)

    def _decompose_task(self, command):
        """Uses LLM to split a complex command into a JSON list of sub-tasks."""
        prompt = f"Decompose the following complex user request into a simple JSON list of discrete sub-tasks that an AI assistant can perform sequentially. Return ONLY a JSON array of strings.\nRequest: {command}"
        response = self._safe_llm_call(prompt)
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            tasks = json.loads(response)
            if isinstance(tasks, list):
                return tasks
        except Exception:
            self.logger.error("Failed to decompose task via LLM, falling back to simple split.")
        
        return [t.strip() for t in command.replace(" and ", " then ").split(" then ") if t.strip()]

    def _execute_single_command(self, command):
        """Deprecated: Replaced by CommandRouter."""
        return self.command_router.run_command(command)

    def _record_lora(self, command: str, response: str):
        """Silently logs interaction to LoRA dataset if module exists."""
        if self.lora_collector is None:
            self.lora_collector = self._init_module("LoRACollector", LoRACollector)
        if self.lora_collector:
            self.lora_collector.log_interaction(command, str(response))

    def check_periodic_alerts(self):
        """Checks for alerts from modules (reminders, temperature, misuse)."""
        alerts = []
        if self.system_monitor:
            alerts.extend(self.system_monitor.check_temp_alerts())
            alerts.extend(self.system_monitor.check_misuse())
        if self.calendar_module:
            alerts.extend(self.calendar_module.check_for_reminders())
        return alerts

    def shutdown(self):
        """Triggers a clean shutdown of the entire application."""
        if self.exit_callback:
            self.logger.info("Shutdown triggered via orchestrator.")
            self.exit_callback()
        else:
            self.logger.warning("Shutdown triggered but no exit_callback is set.")
            import sys
            sys.exit(0)
