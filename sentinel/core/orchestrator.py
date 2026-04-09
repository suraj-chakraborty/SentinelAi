"""
sentinel/core/orchestrator.py
──────────────────────────────
Central hub for all SentinelAI modules.

v2 changes
──────────
  • Wired up SwarmOrchestrator (was always None)
  • Wired up LongTermMemory for cross-session continuity
  • Ollama model pull scheduled in background thread on startup
  • Removed dead methods: _is_complex_command, _decompose_task,
    _execute_single_command  (all replaced by CommandRouter)
  • _init_module now logs exc_info for easier debugging
  • _safe_llm_call: proper logging; Ollama auto-start on failure
  • Proactive loop: reduced default interval to 10 min (configurable)
  • SENTINEL_PROACTIVE_INTERVAL env var controls proactive loop interval
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger("SentinelOrchestrator")


class SentinelOrchestrator:
    def __init__(
        self,
        llm_callback,
        speak_fn=None,
        exit_callback=None,
        master_key=None,
    ):
        self.llm_callback  = llm_callback
        self.speak_fn      = speak_fn or (lambda t, **kw: None)
        self.exit_callback = exit_callback

        # ── Core modules ─────────────────────────────────────────────────────
        from sentinel.modules.system_monitor           import SystemMonitor
        from sentinel.modules.coding                   import CodingModule
        from sentinel.modules.calendar                 import CalendarModule
        from sentinel.modules.web_agent                import WebAgentModule
        from sentinel.modules.vision                   import VisionModule
        from sentinel.modules.knowledge_base           import KnowledgeBaseModule
        from sentinel.modules.iot_hub                  import IoTHubModule
        from sentinel.modules.emotion                  import EmotionModule
        from sentinel.modules.playwright_agent         import PlaywrightAgentModule
        from sentinel.modules.ollama_module            import OllamaModule
        from sentinel.modules.automation_module        import AutomationModule
        from sentinel.modules.gesture_module           import GestureModule
        from sentinel.modules.habit_learner            import HabitLearnerModule
        from sentinel.modules.security_shield          import SecurityShieldModule
        from sentinel.modules.daily_brief              import DailyBriefModule
        from sentinel.modules.lora_collector           import LoRACollector
        from sentinel.modules.quantum_module           import QuantumModule
        from sentinel.utils.notifier                   import WindowsNotifier

        self.system_monitor   = self._init("SystemMonitor",   SystemMonitor)
        self.coding_module    = self._init("CodingModule",    CodingModule)
        self.calendar_module  = self._init("CalendarModule",  CalendarModule)
        self.web_agent        = self._init("WebAgentModule",  WebAgentModule)
        self.vision_module    = self._init("VisionModule",    VisionModule)
        self.knowledge_base   = self._init(
            "KnowledgeBaseModule", KnowledgeBaseModule,
            master_key=master_key, llm_callback=self.llm_callback,
        )
        self.iot_hub          = self._init("IoTHubModule",    IoTHubModule)
        self.emotion_module   = self._init("EmotionModule",   EmotionModule)
        self.playwright_agent = self._init("PlaywrightAgent", PlaywrightAgentModule)
        self.ollama_module    = self._init("OllamaModule",    OllamaModule)
        self.automation_module = self._init("AutomationModule", AutomationModule)
        self.gesture_module   = self._init("GestureModule",   GestureModule)
        self.notifier         = self._init("WindowsNotifier", WindowsNotifier)
        self.file_intel       = self._init("FileIntelligence", self._file_intel_factory)
        self.habit_learner    = self._init("HabitLearner",    self._habit_factory)
        self.security_shield  = self._init("SecurityShield",  self._shield_factory)
        self.daily_brief      = self._init("DailyBrief",      DailyBriefModule, self)
        self.lora_collector   = self._init("LoRACollector",   LoRACollector)
        self.quantum          = self._init("QuantumModule",   QuantumModule)

        # ── Intelligence engines ──────────────────────────────────────────────
        from sentinel.core.agentic_engine  import AgenticEngine
        from sentinel.core.swarm_engine    import SwarmOrchestrator

        self.agentic_engine   = self._init(
            "AgenticEngine", AgenticEngine,
            self, speak_fn=self.speak_fn, gemini_fn=self.llm_callback,
        )
        self.swarm_engine     = self._init(
            "SwarmEngine", SwarmOrchestrator,
            llm_callback=self._safe_llm_call,
        )

        # ── Perception (opt-in, privacy-preserving) ───────────────────────────
        self.perception_stream = None   # enable via enable_perception_stream()

        # ── Long-term memory ──────────────────────────────────────────────────
        from sentinel.memory.long_term_memory import LongTermMemory
        self.long_term_memory = self._init(
            "LongTermMemory", LongTermMemory,
            llm_callback=self._safe_llm_call,
        )

        # ── Computer Use Agent ────────────────────────────────────────────────
        self.computer_use = self._init_computer_use()

        # ── Web server ────────────────────────────────────────────────────────
        from sentinel.core.web_server import SentinelWebServer
        from sentinel.core.jarvis_persona import JarvisPersona
        # Optional: Retrieval module for RAG-style augmentation
        from sentinel.modules.retrieval_module import RetrievalModule
        # New: import for retrieval prompts (Phase 3)
        from sentinel.core.retrieval import RetrievalContext, PromptBuilder
        self.web_server   = self._init("WebServer", SentinelWebServer, self)
        # Phase 1: initialize Jarvis persona (calm default)
        self.jarvis_persona = self._init("JarvisPersona", JarvisPersona, tone="calm")

        # ── Command router ────────────────────────────────────────────────────
        def _router_factory():
            from sentinel.core.command_router import CommandRouter
            return CommandRouter(self)
             
        self.command_router = self._init("CommandRouter", _router_factory)
        # Initialize retrieval module (Phase 3)
        self.retrieval_module = self._init("RetrievalModule", RetrievalModule, knowledge_base=self.knowledge_base)

        # Phase 1: initialize a simple Planner
        from sentinel.core.planner import Planner
        self.planner = self._init("Planner", Planner, self)
        self._jarvis_plan = []
        self._jarvis_progress = 0
        # Phase 5: per-user profiles (in-memory MVP)
        from sentinel.core.user_profiles import UserProfileManager
        self.user_profiles = self._init("UserProfileManager", UserProfileManager)
        self.current_user_id = None

        # ── Start background services ─────────────────────────────────────────
        self._start_background_services()

    # ── Module init helpers ───────────────────────────────────────────────────

    def _init(self, name: str, ctor, *args, **kwargs):
        """Safely instantiate one module; log and return None on failure."""
        try:
            if callable(ctor) and not isinstance(ctor, type):
                return ctor()      # factory function
            return ctor(*args, **kwargs)
        except Exception as exc:
            logger.error("%s init failed: %s", name, exc, exc_info=True)
            return None

    def _init_optional(self, module_path: str, class_name: str, *args, **kwargs):
        """Import a module path and instantiate class_name — silently skip if unavailable."""
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            return cls(*args, **kwargs)
        except Exception as exc:
            logger.debug("Optional module %s.%s not loaded: %s", module_path, class_name, exc)
            return None

    def _file_intel_factory(self):
        if not self.knowledge_base:
            return None
        from sentinel.modules.file_intelligence_module import FileIntelligenceModule
        return FileIntelligenceModule(self.knowledge_base)

    def _habit_factory(self):
        if not self.knowledge_base:
            return None
        from sentinel.modules.habit_learner import HabitLearnerModule
        return HabitLearnerModule(self.knowledge_base)

    def _shield_factory(self):
        if not self.notifier:
            return None
        from sentinel.modules.security_shield import SecurityShieldModule
        return SecurityShieldModule(self.notifier, gemini_fn=self.llm_callback)

    def _init_computer_use(self):
        try:
            from sentinel.core.computer_use_agent import ComputerUseAgent
            return ComputerUseAgent(speak_fn=self.speak_fn)
        except Exception as exc:
            logger.debug("ComputerUseAgent not available: %s", exc)
            return None

    # ── Master key ────────────────────────────────────────────────────────────

    def set_master_key(self, key: bytes) -> None:
        if self.knowledge_base:
            self.knowledge_base.master_key = key
            logger.info("Master key updated for KnowledgeBase.")

    # ── Background services ───────────────────────────────────────────────────

    def _start_background_services(self) -> None:
        services = [
            ("WebServer",    self.web_server,        "start"),
            ("HabitLearner", self.habit_learner,     "start_monitoring"),
            ("SecurityShield", self.security_shield, "start_shield"),
        ]
        for name, obj, method in services:
            if obj and hasattr(obj, method):
                try:
                    getattr(obj, method)()
                except Exception as exc:
                    logger.error("%s.%s() failed: %s", name, method, exc)

        # Pull local Ollama model in background (non-blocking)
        if self.ollama_module:
            threading.Thread(
                target=self._pull_ollama_model, daemon=True
            ).start()

        # Start proactive intelligence loop
        self._start_proactive_loop()

    # ── Jarvis (Phase 2 MVP) ───────────────────────────────────────────────
    def start_jarvis(self, goal: str) -> dict:
        """Initialize a Jarvis planning session for a user-specified goal."""
        if not self.planner:
            return {"error": "Planner not available"}
        plan = self.planner.plan(goal)
        self._jarvis_plan = plan
        self._jarvis_progress = 0
        return {"status": "planned", "steps": [s.text for s in plan]}

    def jarvis_execute_next(self) -> dict:
        """Execute the next step of the current Jarvis plan. This is a minimal MVP path: uses LLM to simulate execution."""
        if not getattr(self, "_jarvis_plan", None) or not self._jarvis_plan:
            return {"status": "no_plan"}
        next_step = self._jarvis_plan[0].text if isinstance(self._jarvis_plan[0], type(self.planner._plan[0])) else str(self._jarvis_plan[0])
        # Use LLM fallback to simulate execution of the step
        if self._jarvis_plan and self._jarvis_plan[0].done:
            self._jarvis_plan.pop(0)
            return {"status": "step_done"}
        # Simulate execution by asking the LLM to interpret the step (if available)
        result = "No execution result"  # default
        try:
            if self and hasattr(self, "_safe_llm_call"):
                result = self._safe_llm_call(f"Execute: {next_step}")
        except Exception:
            result = "Execution simulated"
        self._jarvis_plan[0] = type(self._jarvis_plan[0])(self._jarvis_plan[0].text)  # keep type
        self._jarvis_plan.pop(0)
        self._jarvis_progress += 1
        return {"status": "executed", "step": next_step, "result": result}

    def jarvis_reflect(self) -> dict:
        """Return a light-weight reflection of current Jarvis plan state."""
        remaining = [getattr(s, "text", str(s)) for s in getattr(self, "_jarvis_plan", [])]
        if not remaining:
            return {"status": "complete", "notes": "No remaining steps"}
        return {
            "status": "in_progress",
            "remaining_steps": remaining,
            "progress": getattr(self, "_jarvis_progress", 0)
        }

    def _pull_ollama_model(self) -> None:
        """Start Ollama service and pull the selected model if missing."""
        try:
            if not self.ollama_module:
                return
            self.ollama_module.start_service()
            time.sleep(3)   # give daemon time to start
            self.ollama_module.pull_model_if_missing()
        except Exception as exc:
            logger.warning("Ollama background init failed: %s", exc)

    def enable_perception_stream(self) -> bool:
        """
        Opt-in: start the omnipresent screen perception stream.
        Disabled by default for user privacy.
        """
        if self.perception_stream and self.perception_stream._running:
            return True
        try:
            from sentinel.core.perception_stream import PerceptionStream
            self.perception_stream = PerceptionStream(
                llm_callback=self._safe_vision_call
            )
            return self.perception_stream.start()
        except Exception as exc:
            logger.error("PerceptionStream start failed: %s", exc)
            return False

    def _safe_vision_call(self, prompt: str, image_b64: str = None) -> str:
        """Wrapper that threads image_b64 through the LLM callback if supported."""
        try:
            if image_b64 and hasattr(self.llm_callback, "__call__"):
                # Try kwargs first; Gemini wrapper accepts image_b64
                import inspect
                sig = inspect.signature(self.llm_callback)
                if "image_b64" in sig.parameters:
                    return self.llm_callback(prompt, image_b64=image_b64)
            return self.llm_callback(prompt)
        except Exception as exc:
            logger.debug("Vision LLM call failed: %s", exc)
            return ""

    # ── Proactive loop ────────────────────────────────────────────────────────

    def _start_proactive_loop(self) -> None:
        """Periodic background analysis — vision + reminder checks."""
        if not self.vision_module or not self.notifier:
            return

        interval = int(os.getenv("SENTINEL_PROACTIVE_INTERVAL", "600"))   # default 10 min

        def _loop():
            logger.info("Proactive loop started (interval=%ds).", interval)
            while True:
                try:
                    time.sleep(interval)
                    context = ""
                    if self.perception_stream:
                        try:
                            context = self.perception_stream.get_recent_context(minutes=10)
                        except Exception:
                            pass

                    analysis = self.vision_module.analyze_screen(
                        f"Recent context: {context}\n"
                        "Analyze what the user is doing. If you see a clear opportunity "
                        "to help (error on screen, repetitive task, long article to summarise), "
                        "give a short, specific suggestion. Otherwise return 'NORMAL'."
                    )

                    if (analysis
                            and "NORMAL" not in analysis.upper()
                            and len(analysis) > 10
                            and "API key" not in analysis
                            and "trouble connecting" not in analysis):
                        self.notifier.show_notification("Sentinel Suggestion", analysis[:200])
                        logger.info("Proactive suggestion: %s", analysis[:80])

                except Exception as exc:
                    logger.error("Proactive loop error: %s", exc)

        threading.Thread(target=_loop, daemon=True).start()

    # ── LLM gateway ──────────────────────────────────────────────────────────

    def _safe_llm_call(self, prompt: str) -> str:
        """End-to-end LLM call with memory grounding and per-user persona (Phase 5 MVP)."""
        # Build a robust prompt with persona override and fallbacks
        try:
            from sentinel.core.retrieval import RetrievalContext, PromptBuilder
            memory_context = self.long_term_memory.inject_past_context(prompt) if self.long_term_memory else ""
            kb_context = self.knowledge_base.query_knowledge(prompt) if getattr(self, 'knowledge_base', None) and hasattr(self.knowledge_base, 'query_knowledge') else ""
            retrieval_context = self.retrieval_module.retrieve(prompt) if getattr(self, 'retrieval_module', None) else ""
            persona_str = ""
            if getattr(self, 'jarvis_persona', None) and self.jarvis_persona is not None:
                try:
                    persona_str = self.jarvis_persona.get_prompt_schip()
                except Exception:
                    persona_str = ""
            # Phase 5.2: apply per-user persona if a profile exists for the active user
            persona_override = persona_str
            if getattr(self, 'current_user_id', None) and getattr(self, 'user_profiles', None):
                try:
                    profile = self.user_profiles.get_profile(self.current_user_id)
                    if profile and getattr(profile, 'persona', None):
                        persona_override = profile.persona
                except Exception:
                    pass
            rc = RetrievalContext(memory_context=memory_context, retrieved_context=retrieval_context, knowledge_context=kb_context, current_task=prompt, persona=persona_override)
            full_prompt = PromptBuilder(rc).build()
        except Exception:
            # Fallback to legacy simple composition
            memory_context = self.long_term_memory.inject_past_context(prompt) if self.long_term_memory else ""
            kb_context = self.knowledge_base.query_knowledge(prompt) if getattr(self, 'knowledge_base', None) and hasattr(self.knowledge_base, 'query_knowledge') else ""
            retrieval_context = self.retrieval_module.retrieve(prompt) if getattr(self, 'retrieval_module', None) else ""
            parts = []
            if memory_context:
                parts.append(memory_context)
            if isinstance(retrieval_context, str) and retrieval_context.strip():
                parts.append("[Retrieved]\n" + retrieval_context.strip())
            if isinstance(kb_context, str) and kb_context.strip():
                parts.append("[Knowledge]" + ("\n" if kb_context[:1] != "[" else "") + kb_context.strip())
            parts.append(f"[Current Task]\n{prompt}")
            full_prompt = "\n".join(parts) if parts else prompt

        try:
            llm_res = self.llm_callback(full_prompt)
            if not hasattr(self, "_jarvis_transcript"):
                self._jarvis_transcript = []
            self._jarvis_transcript.append({"step": getattr(self, 'last_step_text', 'JarvisStep'), "result": llm_res, "timestamp": time.time()})
            return llm_res
        except Exception as exc:
            logger.warning("Online LLM failed (%s) — fallback to Ollama.", exc)
            if self.ollama_module:
                if not self.ollama_module.is_available():
                    self.ollama_module.start_service()
                    time.sleep(2)
                if self.ollama_module.is_available():
                    return self.ollama_module.generate(full_prompt)
            return "Both online and offline AI services are currently unavailable."

    # ── Command dispatch ──────────────────────────────────────────────────────

    def run_command(self, command: str) -> str:
        """Route command through the NLP command engine."""
        if not command:
            return ""
        logger.info("Orchestrating: %s", command[:80])
        
        # Track memory triggers
        if self.long_term_memory:
             # We should theoretically have a ConversationManager instance here,
             # but for now we manually signal a turn occurred.
            self.long_term_memory._turns_since_last_summary += 1

        if not self.command_router:
            logger.error("CommandRouter not available — falling back to AI agent.")
            return self.ai_fallback(command)
            
        result = self.command_router.run_command(command)
        if result:
            self._record_lora(command, result)
            return result
        return "I'm sorry, I couldn't process that command."

    # ── LoRA data collection ──────────────────────────────────────────────────

    def _record_lora(self, command: str, response: str) -> None:
        if self.lora_collector:
            try:
                self.lora_collector.log_interaction(command, str(response))
            except Exception as exc:
                logger.debug("LoRA log failed: %s", exc)

    # ── Periodic alert checks ─────────────────────────────────────────────────

    def check_periodic_alerts(self) -> list:
        alerts = []
        if self.system_monitor:
            try:
                alerts += self.system_monitor.check_temp_alerts()
                alerts += self.system_monitor.check_misuse()
            except Exception:
                pass
        if self.calendar_module:
            try:
                alerts += self.calendar_module.check_for_reminders()
            except Exception:
                pass
        return alerts

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        logger.info("Orchestrator shutdown requested.")
        if self.exit_callback:
            self.exit_callback()
        else:
            import sys
            sys.exit(0)

    # ── Legacy alias kept for backwards compatibility ─────────────────────────

    @property
    def web_agent_module(self):
        return self.web_agent

    @web_agent_module.setter
    def web_agent_module(self, val):
        self.web_agent = val
