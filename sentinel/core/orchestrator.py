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
    from sentinel.core.plugin_system import PluginSystem
    _PLUGIN_SYSTEM = True
except ImportError:
    _PLUGIN_SYSTEM = False

try:
    from sentinel.security.sandbox import Sandbox
    _SANDBOX = True
except ImportError:
    _SANDBOX = False

class SentinelOrchestrator:
    def __init__(self, llm_callback, master_key=None):
        self.llm_callback = llm_callback
        self.logger = logging.getLogger("SentinelOrchestrator")

        # ── Core Modules ──────────────────────────────────────────────────
        self.system_monitor    = self._init_module("SystemMonitor", SystemMonitor)
        self.coding_module     = self._init_module("CodingModule", CodingModule)
        self.calendar_module   = self._init_module("CalendarModule", CalendarModule)
        self.web_agent_module  = self._init_module("WebAgentModule", WebAgentModule)
        self.vision_module     = self._init_module("VisionModule", VisionModule)
        self.knowledge_base    = self._init_module("KnowledgeBaseModule", KnowledgeBaseModule, master_key=master_key)
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
        self.agentic_engine    = self._init_module("AgenticEngine", AgenticEngine, self)
        self.swarm_engine      = None
        self.perception_stream = None
        self.lora_collector    = None
        self.web_server        = self._init_module("SentinelWebServer", SentinelWebServer, self)

        # ── Tier 3 Modules ────────────────────────────────────────────────
        self.computer_use = None
        self.plugin_system = None

        self.sandbox = None
        if _SANDBOX:
            self.sandbox = Sandbox

        # ── Sci-Fi Modules ────────────────────────────────────────────────
        self.robotics = self._init_module("RoboticsModule", RoboticsModule)
        self.bci = self._init_module("BCIModule", BCIModule)
        self.quantum = self._init_module("QuantumModule", QuantumModule)
        self.proxy_shield = None
        
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
            self.logger.info("Autonomous Cron loop started.")
            while True:
                try:
                    time.sleep(300) 
                    self.logger.info("Running proactive context analysis...")
                    
                    # Merge static screen analysis with multi-modal perception stream
                    context = ""
                    if self.perception_stream:
                        context = self.perception_stream.get_recent_context(minutes=15)

                    analysis = self.vision_module.analyze_screen(
                        f"Recent Context: {context}\nAnalyze the user's current activity. If you see a way to help (e.g., debugging a visible error, summarizing a long page, or automating a repetitive task), provide a short, proactive suggestion. If everything looks normal, return 'NORMAL'."
                    )
                    
                    if "NORMAL" not in analysis.upper() and len(analysis) > 5:
                        self.notifier.show_notification("Sentinel Proactive Alert", analysis)
                        self.logger.info(f"Proactive suggestion sent: {analysis}")
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
        """Dispatches a command through plugin system → intent router → agentic engine → LLM."""
        cmd = (command or "").lower().strip()
        self.logger.info(f"Running command: {cmd}")

        # 0. Plugin system (first priority — community extensions)
        if self.plugin_system is None and _PLUGIN_SYSTEM:
            self.plugin_system = self._init_module("PluginSystem", PluginSystem, self)
            if self.plugin_system:
                count = self.plugin_system.load_all()
                self.logger.info(f"Plugin system loaded {count} plugin(s).")
        if self.plugin_system:
            plugin_result = self.plugin_system.dispatch(command)
            if plugin_result:
                self._record_lora(command, plugin_result)
                return plugin_result

        # Perception Stream Toggles
        if "start perception" in cmd or "turn on awareness" in cmd:
            stream = self._ensure_optional_module("perception_stream", "PerceptionStream", PerceptionStream, self._safe_llm_call)
            if stream and stream.start():
                ans = "Omnipresent Perception Stream enabled. I am now watching the screen."
                self._record_lora(command, ans)
                return ans
            return "Perception stream unavailable."
        
        if "stop perception" in cmd or "turn off awareness" in cmd:
            if self.perception_stream:
                self.perception_stream.stop()
            ans = "Omnipresent Perception Stream disabled."
            self._record_lora(command, ans)
            return ans

        # Phase 5: Swarm Delegation
        if "swarm" in cmd or "delegate task" in cmd or "multi-agent" in cmd:
            swarm_engine = self._ensure_optional_module("swarm_engine", "SwarmOrchestrator", SwarmOrchestrator, self._safe_llm_call)
            if not swarm_engine:
                return "Swarm engine unavailable."
            goal = command.replace("swarm", "").strip()
            ans = swarm_engine.run_swarm(goal)
            self._record_lora(command, ans)
            return ans

        # ── Sci-Fi Tier Routing ──────────────────────────────────────────────
        
        # 1. Embodied AI
        if "robot" in cmd or "automata" in cmd or "rover" in cmd:
            robotics = self._ensure_optional_module("robotics", "RoboticsModule", RoboticsModule)
            if not robotics:
                return "Robotics module unavailable."
            ans = robotics.instruct_automata(command, self._safe_llm_call)
            self._record_lora(command, ans)
            return ans
            
        # 2. BCI Activation
        if "connect headset" in cmd or "start bci" in cmd:
            bci = self._ensure_optional_module("bci", "BCIModule", BCIModule, self)
            if not bci:
                return "BCI module unavailable."
            ans = bci.start()
            self._record_lora(command, ans)
            return ans
            
        # 3. Proxy Cyber Shield
        if "enable proxy shield" in cmd or "start zero trust" in cmd:
            proxy_shield = self._ensure_optional_module("proxy_shield", "ProxyShield", ProxyShield, self._safe_llm_call)
            if not proxy_shield:
                return "Proxy shield unavailable."
            import asyncio
            asyncio.run_coroutine_threadsafe(proxy_shield.start(), asyncio.get_event_loop())
            ans = "Deep inspection LLM proxy started on port 8080."
            self._record_lora(command, ans)
            return ans
            
        # 4. Quantum Connect
        if "quantum optimization" in cmd or "calculate using qpu" in cmd:
            quantum = self._ensure_optional_module("quantum", "QuantumModule", QuantumModule)
            if not quantum:
                return "Quantum module unavailable."
            ans = quantum.solve_optimization(command, self._safe_llm_call)
            self._record_lora(command, ans)
            return ans

        # 1. Agentic Goal Handling
        if "achieve goal" in cmd or "autonomous task" in cmd or "computer use" in cmd:
            # Computer Use Agent (vision-guided desktop control)
            if "computer use" in cmd and _COMPUTER_USE:
                self.computer_use = self._ensure_optional_module("computer_use", "ComputerUseAgent", ComputerUseAgent)
            if "computer use" in cmd and self.computer_use:
                goal = command.split("computer use", 1)[-1].strip()
                ans = self.computer_use.execute(goal or command)
                self._record_lora(command, ans)
                return ans
            # ReAct Agentic Engine
            if self.agentic_engine:
                goal = command.split("goal", 1)[-1].strip() if "goal" in cmd else command
                ans = self.agentic_engine.execute_autonomous_goal(goal)
                self._record_lora(command, ans)
                return ans
            return "Agentic engine unavailable."

        # 2. Sandboxed code execution
        if ("run this code" in cmd or "execute code" in cmd or "run python" in cmd) and self.sandbox:
            code_part = command.split(":", 1)[-1].strip() if ":" in command else command
            success, output = self.sandbox.execute(code_part)
            self._record_lora(command, output)
            return output

        # 2. Daily Brief
        if "morning brief" in cmd or "daily brief" in cmd or "status report" in cmd:
            if not self.daily_brief:
                return "Daily brief module unavailable."
            ans = self.daily_brief.generate_brief()
            self._record_lora(command, ans)
            return ans

        # 3. Gesture Control
        if "start gesture control" in cmd or "enable gesture" in cmd:
            if not self.gesture_module:
                return "Gesture module unavailable."
            self.gesture_module.start()
            return "Gesture control enabled. Move index finger to move mouse, fist to press Enter."
        if "stop gesture control" in cmd or "disable gesture" in cmd:
            if not self.gesture_module:
                return "Gesture module unavailable."
            self.gesture_module.stop()
            return "Gesture control disabled."

        # Task Decomposition for complex queries
        if self._is_complex_command(cmd):
            if self.notifier:
                self.notifier.show_notification("SentinelAI", "Breaking down complex task...")
            sub_tasks = self._decompose_task(command)
            results = []
            for i, task in enumerate(sub_tasks, 1):
                self.logger.info(f"Executing sub-task {i}: {task}")
                if self.notifier:
                    self.notifier.show_notification("SentinelAI", f"Step {i}/{len(sub_tasks)}: {task[:40]}...")
                results.append(self._execute_single_command(task))
            
            summary_prompt = f"The user asked: {command}\n\nI performed the following sub-tasks and got these results:\n"
            for t, r in zip(sub_tasks, results):
                summary_prompt += f"Task: {t}\nResult: {r}\n\n"
            summary_prompt += "Please provide a final summary of what was accomplished."
            return self._safe_llm_call(summary_prompt)
        
        return self._execute_single_command(command)

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
        """The original logic for executing a single atomic command."""
        cmd = (command or "").lower().strip()
        
        # File Intelligence
        if "summarize my recent files" in cmd or "scan documents" in cmd:
            if not self.file_intel:
                return "File intelligence module unavailable."
            downloads = os.path.join(os.path.expanduser("~"), "Downloads")
            return self.file_intel.summarize_recent_files(downloads, hours=24, llm_callback=self._safe_llm_call)
        if "what was in the pdf" in cmd or "search document" in cmd:
            if not self.knowledge_base:
                return "Knowledge base unavailable."
            query = command.split("pdf", 1)[1].strip() if "pdf" in cmd else command
            return self.knowledge_base.query_knowledge(f"About file content: {query}")

        # Cross-App Automation (GUI control)
        if "move mouse" in cmd or "click" in cmd or "type text" in cmd or "press key" in cmd:
            if not self.automation_module:
                return "Automation module unavailable."
            return self.automation_module.execute_gui_task(command, self._safe_llm_call)

        # Personal Knowledge Base (RAG)
        if "remember this" in cmd or "save to knowledge" in cmd or "add to memory" in cmd:
            if not self.knowledge_base:
                return "Knowledge base unavailable."
            info = command.split("this", 1)[1].strip() if "this" in cmd else command
            success, msg = self.knowledge_base.add_information(info)
            return msg
        if "what do you know about" in cmd or "search memory" in cmd or "find info" in cmd:
            if not self.knowledge_base:
                return "Knowledge base unavailable."
            query = command.split("about", 1)[1].strip() if "about" in cmd else command
            return self.knowledge_base.query_knowledge(query)

        # IoT Control
        if "turn on" in cmd or "turn off" in cmd or "control light" in cmd or "run scene" in cmd:
            if not self.iot_hub:
                return "IoT module unavailable."
            entity = cmd.replace("turn on", "").replace("turn off", "").strip()
            service = "turn_on" if "on" in cmd else "turn_off"
            success, msg = self.iot_hub.control_home_assistant(entity, service=service)
            return msg

        # Playwright Autonomous Browser
        if "book" in cmd or "order" in cmd or "autonomous search" in cmd or "browse" in cmd:
            if not self.playwright_agent:
                return "Playwright agent unavailable."
            return self.playwright_agent.perform_autonomous_task(command)

        # Vision/Screen perception
        if "what is on my screen" in cmd or "analyze screen" in cmd or "describe window" in cmd:
            if not self.vision_module:
                return "Vision module unavailable."
            return self.vision_module.analyze_screen(command)

        # Quantum Optimization Fallback
        if "quantum optimization" in cmd or "calculate using qpu" in cmd:
            if not self.quantum:
                return "Quantum module unavailable."
            return self.quantum.solve_optimization(command)

        # Coding commands
        if "write code" in cmd or "run code" in cmd or "python script" in cmd:
            if not self.coding_module:
                return "Coding module unavailable."
            return self.coding_module.generate_and_execute(command, self._safe_llm_call)

        # Calendar/Meet commands
        if "upcoming meets" in cmd or "google meet" in cmd or "calendar" in cmd:
            if not self.calendar_module:
                return "Calendar module unavailable."
            meets = self.calendar_module.get_upcoming_meets()
            if not meets:
                return "No upcoming Meets found."
            response = "Upcoming Meets:\n"
            for meet in meets:
                response += f"- {meet['summary']} at {meet['start']}\n"
            return response

        # System Monitor commands
        if "system status" in cmd or "temperature" in cmd or "cpu temp" in cmd:
            if not self.system_monitor:
                return "System monitor unavailable."
            cpu_temp = self.system_monitor.get_cpu_temp()
            gpu_temp = self.system_monitor.get_gpu_temp()
            status = f"CPU Temp: {cpu_temp if cpu_temp else 'N/A'}°C, GPU Temp: {gpu_temp if gpu_temp else 'N/A'}°C"
            return status

        # Web Agent commands (booking, search, etc.)
        if "book ticket" in cmd or "search for" in cmd or "find" in cmd:
            if not self.web_agent_module:
                return "Web agent module unavailable."
            return self.web_agent_module.book_ticket_generic(command)

        # Fallback: Only use LLM if explicitly requested or if it's a question
        ai_triggers = ["ai", "gemini", "explain", "why", "how", "what is", "who is", "search for", "think"]
        if any(k in cmd for k in ai_triggers) or "?" in cmd:
            final_ans = self._safe_llm_call(command)
            self._record_lora(command, final_ans)
            return final_ans
            
        return None # Return None to let the caller know it wasn't handled

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
