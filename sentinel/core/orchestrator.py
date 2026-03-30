import os
import logging
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
from sentinel.core.agentic_engine import AgenticEngine
from sentinel.utils.notifier import WindowsNotifier
from sentinel.core.web_server import SentinelWebServer
import json
import time
import threading

class SentinelOrchestrator:
    def __init__(self, llm_callback, master_key=None):
        self.llm_callback = llm_callback
        self.logger = logging.getLogger("SentinelOrchestrator")
        
        # Initialize modules with error handling to prevent startup crashes
        try:
            self.system_monitor = SystemMonitor()
            self.coding_module = CodingModule()
            self.calendar_module = CalendarModule()
            self.web_agent_module = WebAgentModule()
            self.vision_module = VisionModule()
            self.knowledge_base = KnowledgeBaseModule(master_key=master_key)
            self.iot_hub = IoTHubModule()
            self.emotion_module = EmotionModule()
            self.playwright_agent = PlaywrightAgentModule()
            self.ollama_module = OllamaModule()
            self.automation_module = AutomationModule()
            self.gesture_module = GestureModule()
            self.file_intel = FileIntelligenceModule(self.knowledge_base)
            self.habit_learner = HabitLearnerModule(self.knowledge_base)
            self.security_shield = SecurityShieldModule(WindowsNotifier())
            self.daily_brief = DailyBriefModule(self)
            self.agentic_engine = AgenticEngine(self)
            self.notifier = WindowsNotifier()
            self.web_server = SentinelWebServer(self)
        except Exception as e:
            self.logger.critical(f"Critical module initialization failure: {e}")
            # Ensure at least basic modules exist if possible, or raise if truly critical
            raise e
        
        # Start background services
        try:
            self.web_server.start()
            self.habit_learner.start_monitoring()
            self.security_shield.start_shield()
            self._start_proactive_loop()
        except Exception as e:
            self.logger.error(f"Failed to start background services: {e}")

    def set_master_key(self, key):
        """Updates the master key for encrypted modules."""
        self.knowledge_base.master_key = key
        self.logger.info("Master key updated for Knowledge Base.")

    def _start_proactive_loop(self):
        """Starts the proactive intelligence loop in a background thread."""
        def _loop():
            self.logger.info("Proactive Intelligence loop started.")
            while True:
                try:
                    # Every 5 minutes, proactively analyze context
                    time.sleep(300) 
                    self.logger.info("Running proactive context analysis...")
                    analysis = self.vision_module.analyze_screen(
                        "Analyze the user's current activity. If you see a way to help (e.g., debugging a visible error, summarizing a long page, or automating a repetitive task), provide a short, one-sentence proactive suggestion. If everything looks normal, return 'NORMAL'."
                    )
                    
                    if "NORMAL" not in analysis.upper() and len(analysis) > 5:
                        self.notifier.show_notification("Sentinel Proactive Idea", analysis)
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
            if self.ollama_module.is_available():
                return self.ollama_module.generate(prompt)
            return "Both online and offline LLM services are currently unavailable."

    def run_command(self, command):
        """Dispatches a command, handles agentic goals and task decomposition."""
        cmd = (command or "").lower().strip()
        self.logger.info(f"Running command: {cmd}")

        # 1. Agentic Goal Handling (e.g., "Sentinel, achieve goal: ...")
        if "achieve goal" in cmd or "autonomous task" in cmd:
            goal = command.split("goal", 1)[1].strip()
            return self.agentic_engine.execute_autonomous_goal(goal)

        # 2. Daily Brief
        if "morning brief" in cmd or "daily brief" in cmd or "status report" in cmd:
            return self.daily_brief.generate_brief()

        # 3. Gesture Control
        if "start gesture control" in cmd or "enable gesture" in cmd:
            self.gesture_module.start()
            return "Gesture control enabled. Move index finger to move mouse, fist to press Enter."
        if "stop gesture control" in cmd or "disable gesture" in cmd:
            self.gesture_module.stop()
            return "Gesture control disabled."

        # Task Decomposition for complex queries
        if self._is_complex_command(cmd):
            self.notifier.show_notification("SentinelAI", "Breaking down complex task...")
            sub_tasks = self._decompose_task(command)
            results = []
            for i, task in enumerate(sub_tasks, 1):
                self.logger.info(f"Executing sub-task {i}: {task}")
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
            downloads = os.path.join(os.path.expanduser("~"), "Downloads")
            return self.file_intel.summarize_recent_files(downloads, hours=24, llm_callback=self._safe_llm_call)
        if "what was in the pdf" in cmd or "search document" in cmd:
            query = command.split("pdf", 1)[1].strip() if "pdf" in cmd else command
            return self.knowledge_base.query_knowledge(f"About file content: {query}")

        # Cross-App Automation (GUI control)
        if "move mouse" in cmd or "click" in cmd or "type text" in cmd or "press key" in cmd:
            return self.automation_module.execute_gui_task(command, self._safe_llm_call)

        # Personal Knowledge Base (RAG)
        if "remember this" in cmd or "save to knowledge" in cmd or "add to memory" in cmd:
            info = command.split("this", 1)[1].strip() if "this" in cmd else command
            success, msg = self.knowledge_base.add_information(info)
            return msg
        if "what do you know about" in cmd or "search memory" in cmd or "find info" in cmd:
            query = command.split("about", 1)[1].strip() if "about" in cmd else command
            return self.knowledge_base.query_knowledge(query)

        # IoT Control
        if "turn on" in cmd or "turn off" in cmd or "control light" in cmd or "run scene" in cmd:
            entity = cmd.replace("turn on", "").replace("turn off", "").strip()
            service = "turn_on" if "on" in cmd else "turn_off"
            success, msg = self.iot_hub.control_home_assistant(entity, service=service)
            return msg

        # Playwright Autonomous Browser
        if "book" in cmd or "order" in cmd or "autonomous search" in cmd:
            return self.playwright_agent.perform_autonomous_task(command, self._safe_llm_call)

        # Vision/Screen perception
        if "what is on my screen" in cmd or "analyze screen" in cmd or "describe window" in cmd:
            return self.vision_module.analyze_screen(command)

        # Coding commands
        if "write code" in cmd or "run code" in cmd or "python script" in cmd:
            return self.coding_module.generate_and_execute(command, self._safe_llm_call)

        # Calendar/Meet commands
        if "upcoming meets" in cmd or "google meet" in cmd or "calendar" in cmd:
            meets = self.calendar_module.get_upcoming_meets()
            if not meets:
                return "No upcoming Meets found."
            response = "Upcoming Meets:\n"
            for meet in meets:
                response += f"- {meet['summary']} at {meet['start']}\n"
            return response

        # System Monitor commands
        if "system status" in cmd or "temperature" in cmd or "cpu temp" in cmd:
            cpu_temp = self.system_monitor.get_cpu_temp()
            gpu_temp = self.system_monitor.get_gpu_temp()
            status = f"CPU Temp: {cpu_temp if cpu_temp else 'N/A'}°C, GPU Temp: {gpu_temp if gpu_temp else 'N/A'}°C"
            return status

        # Web Agent commands (booking, search, etc.)
        if "book ticket" in cmd or "search for" in cmd or "find" in cmd:
            return self.web_agent_module.book_ticket_generic(command)

        # Fallback to LLM for general conversation or unhandled tasks
        return self._safe_llm_call(command)

    def check_periodic_alerts(self):
        """Checks for alerts from modules (reminders, temperature, misuse)."""
        alerts = []
        alerts.extend(self.system_monitor.check_temp_alerts())
        alerts.extend(self.system_monitor.check_misuse())
        alerts.extend(self.calendar_module.check_for_reminders())
        return alerts
