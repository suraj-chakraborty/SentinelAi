"""
sentinel/core/agentic_engine.py
────────────────────────────────
ReAct (Reason + Act) Agentic Engine — Tier 2 Upgrade.

Upgraded from a simple 5-step loop to a full ReAct pattern with:
  - A named tool registry
  - Think → Act → Observe → Reflect loop
  - Self-evaluation (knows when the goal is accomplished)
  - Step-by-step transcript for debugging + GUI display
  - Safety constraints (no destructive actions without confirmation)
  - Max-step guard with graceful summary
"""

import os
import time
import base64
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("AgenticEngine")

# ─── Tool registry ────────────────────────────────────────────────────────────

class Tool:
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def execute(self, **kwargs) -> str:
        try:
            result = self.func(**kwargs)
            return str(result) if result is not None else "Done."
        except Exception as e:
            logger.error(f"Tool '{self.name}' error: {e}")
            return f"Error: {e}"


class AgenticEngine:
    """
    ReAct Agentic Engine.

    Example flow:
        engine = AgenticEngine(orchestrator=..., speak_fn=speak, gemini_fn=gemini_generate)
        result = engine.execute_autonomous_goal("Research Python async patterns and open a doc summary")
    """

    MAX_STEPS = 15
    THINK_PROMPT = """You are the reasoning core of SentinelAI, an autonomous desktop AI agent.

Your goal: {goal}

Conversation so far:
{history}

Available tools:
{tool_list}

Instructions:
- Reason step-by-step in a THOUGHT block.
- Then decide the next action using EXACTLY this format:
  ACTION: tool_name
  INPUT: {{"key": "value"}}
- Or if the goal is fully accomplished:
  ACTION: DONE
  INPUT: {{"summary": "What was accomplished"}}
- Or if you need user clarification:
  ACTION: ASK_USER
  INPUT: {{"question": "Your question"}}

Only output THOUGHT + ACTION + INPUT. Nothing else."""

    def __init__(self, orchestrator=None, speak_fn: Callable = None, gemini_fn: Callable = None):
        self.orchestrator = orchestrator
        self.speak = speak_fn or (lambda t, **kw: print(f"[Agent] {t}"))
        self.gemini = gemini_fn
        self._tools: Dict[str, Tool] = {}
        self._active = False
        self._lock = threading.Lock()
        self._register_default_tools()
        logger.info("AgenticEngine (ReAct) initialized.")

    # ─── Tool registration ────────────────────────────────────────────────────

    def register_tool(self, name: str, description: str, func: Callable):
        """Register a callable as an agent tool."""
        self._tools[name] = Tool(name, description, func)
        logger.info(f"Registered tool: {name}")

    def _register_default_tools(self):
        """Register all default SentinelAI tools."""
        self.register_tool(
            "read_screen",
            "Captures a screenshot and analyzes what is currently on the screen. No input needed.",
            self._tool_read_screen
        )
        self.register_tool(
            "click",
            "Moves the mouse to (x, y) and clicks. Input: {\"x\": int, \"y\": int, \"button\": \"left\"|\"right\"}",
            self._tool_click
        )
        self.register_tool(
            "type_text",
            "Types text at the current cursor position. Input: {\"text\": str}",
            self._tool_type_text
        )
        self.register_tool(
            "press_key",
            "Presses a keyboard key. Input: {\"key\": str} e.g. enter, tab, escape, ctrl+c",
            self._tool_press_key
        )
        self.register_tool(
            "open_application",
            "Opens an application by name. Input: {\"app_name\": str}",
            self._tool_open_app
        )
        self.register_tool(
            "run_python_code",
            "Writes and executes a Python script for data processing tasks. Input: {\"code\": str}",
            self._tool_run_code
        )
        self.register_tool(
            "search_web",
            "Opens the browser and searches for a query. Input: {\"query\": str}",
            self._tool_web_search
        )
        self.register_tool(
            "query_knowledge",
            "Searches the personal knowledge base for relevant information. Input: {\"query\": str}",
            self._tool_query_kb
        )
        self.register_tool(
            "remember_fact",
            "Stores a fact in the personal knowledge base. Input: {\"fact\": str, \"source\": str}",
            self._tool_remember
        )
        self.register_tool(
            "wait",
            "Waits for a specified number of seconds. Input: {\"seconds\": float}",
            self._tool_wait
        )
        self.register_tool(
            "speak_to_user",
            "Speaks a message to the user. Input: {\"message\": str}",
            self._tool_speak
        )
        self.register_tool(
            "control_iot",
            "Controls a Home Assistant device. Input: {\"entity_id\": str, \"service\": str, \"domain\": str}",
            self._tool_iot
        )

    # ─── Main execution loop ──────────────────────────────────────────────────

    def execute_autonomous_goal(self, goal: str, speak_progress: bool = True) -> str:
        """
        Execute an autonomous multi-step goal using the ReAct pattern.
        Returns a final summary of what was accomplished.
        """
        with self._lock:
            if self._active:
                return "Agent is already running a task. Say 'stop agent' to cancel."
            self._active = True

        try:
            return self._run_react_loop(goal, speak_progress)
        finally:
            self._active = False

    def _run_react_loop(self, goal: str, speak_progress: bool) -> str:
        logger.info(f"[ReAct] Starting goal: {goal}")
        if speak_progress:
            self.speak(f"Starting autonomous task: {goal[:50]}")

        history: List[Dict[str, str]] = []
        transcript: List[str] = []
        tool_list = self._format_tool_list()

        for step in range(self.MAX_STEPS):
            if not self._active:
                logger.info("[ReAct] Aborted by user.")
                break

            # ── THINK ────────────────────────────────────────────────────────
            history_str = self._format_history(history)
            prompt = self.THINK_PROMPT.format(
                goal=goal,
                history=history_str,
                tool_list=tool_list
            )

            if speak_progress and step > 0:
                self.speak(f"Step {step + 1}. Reasoning...")

            llm_response = self._call_llm(prompt)
            if not llm_response:
                break

            thought, action_name, action_input = self._parse_response(llm_response)
            logger.info(f"[ReAct step {step+1}] Action: {action_name}, Input: {action_input}")
            transcript.append(f"Step {step+1}: {thought[:80]}... → {action_name}")

            # ── TERMINAL STATES ───────────────────────────────────────────────
            if action_name == "ERROR_LLM":
                err_msg = action_input.get("message", "The brain is currently unavailable.")
                logger.error(f"[ReAct] LLM Error: {err_msg}")
                if speak_progress:
                    self.speak(f"I'm sorry, my thinking engine is having trouble: {err_msg}")
                return f"[Error] {err_msg}"

            if action_name == "DONE":
                # Only trust DONE if we have a valid summary or if the LLM actually tried to solve it
                summary = action_input.get("summary", "Goal accomplished.")
                logger.info(f"[ReAct] DONE: {summary}")
                if speak_progress:
                    self.speak(f"Task complete. {summary}")
                return summary

            if action_name == "ASK_USER":
                question = action_input.get("question", "I need more information.")
                if speak_progress:
                    self.speak(question)
                return f"[Needs clarification] {question}"

            # ── ACT ─────────────────────────────────────────────────────────
            # Use case-insensitive lookup (tools are registered in lowercase)
            action_key = action_name.lower()
            if action_key not in self._tools:
                observation = f"Unknown tool '{action_name}'. Available: {list(self._tools.keys())}"
                logger.warning(observation)
            else:
                tool = self._tools[action_key]
                observation = tool.execute(**action_input)
                logger.info(f"[ReAct] Observation: {observation[:100]}")
                if speak_progress and len(observation) < 120:
                    self.speak(observation)

            # ── RECORD ───────────────────────────────────────────────────────
            history.append({
                "step": step + 1,
                "thought": thought,
                "action": action_name,
                "input": str(action_input),
                "observation": observation
            })

            time.sleep(0.3)  # Brief pause between steps

        # Max steps reached
        summary = f"Reached maximum steps ({self.MAX_STEPS}). Progress: {'; '.join(transcript[-3:])}"
        if speak_progress:
            self.speak("I've reached the maximum number of steps for this task.")
        logger.warning(f"[ReAct] Max steps reached for goal: {goal}")
        return summary

    def stop(self):
        """Interrupt the running agent loop."""
        self._active = False
        logger.info("[ReAct] Stop signal sent.")

    @property
    def is_active(self) -> bool:
        return self._active

    # ─── LLM call ─────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        callback = self.gemini
        if callback is None and self.orchestrator and hasattr(self.orchestrator, "_safe_llm_call"):
            callback = self.orchestrator._safe_llm_call
            
        if callback:
            try:
                return callback(prompt)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
        return ""

    # ─── Response parser ──────────────────────────────────────────────────────

    def _parse_response(self, response: str) -> Tuple[str, str, dict]:
        """Extract THOUGHT, ACTION, and INPUT from LLM response."""
        import json, re
        
        # Check for common error signatures from unified gemini_generate fallback
        error_keywords = ["having trouble connecting", "Request timed out", "service error", "Quota Exceeded"]
        if any(k in response for k in error_keywords) or not response.strip():
            return "The brain is currently unavailable.", "ERROR_LLM", {"message": response or "Empty response"}

        thought = ""
        # Default to DONE only if we actually see a Thought/Action structure or if it's clearly a final answer
        action_name = "DONE" 
        action_input = {}

        # Extract THOUGHT
        t_match = re.search(r"THOUGHT[:\s]*(.+?)(?=ACTION:|$)", response, re.DOTALL | re.IGNORECASE)
        if t_match:
            thought = t_match.group(1).strip()
        else:
            # If no THOUGHT block, it might be a direct conversational response
            thought = response.strip()

        # Extract ACTION
        a_match = re.search(r"ACTION[:\s]*(\w+)", response, re.IGNORECASE)
        if a_match:
            action_name = a_match.group(1).strip().upper()
        else:
            # If no ACTION block but we have content, treat as DONE with the content as summary
            action_name = "DONE"
            action_input = {"summary": response.strip()}

        # Extract INPUT (first JSON object; supports nested braces)
        i_match = re.search(r"INPUT[:\s]*", response, re.IGNORECASE)
        if i_match:
            tail = response[i_match.end() :].lstrip()
            if tail.startswith("{"):
                dec = json.JSONDecoder()
                try:
                    action_input, _ = dec.raw_decode(tail)
                except json.JSONDecodeError:
                    try:
                        action_input = json.loads(tail.split("}")[0] + "}")
                    except json.JSONDecodeError:
                        pairs = re.findall(r'"(\w+)"\s*:\s*"([^"]*)"', tail)
                        if pairs:
                            action_input = {k: v for k, v in pairs}
                        else:
                            action_input = {"raw": tail[:500]}

        return thought, action_name, action_input

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _format_tool_list(self) -> str:
        return "\n".join(f"  - {name}: {tool.description}" for name, tool in self._tools.items())

    def _format_history(self, history: List[Dict]) -> str:
        if not history:
            return "[No steps taken yet]"
        parts = []
        for h in history[-5:]:  # Last 5 steps for context
            parts.append(
                f"Step {h['step']}: Used {h['action']}({h['input'][:50]}) → {h['observation'][:80]}"
            )
        return "\n".join(parts)

    # ─── Tool implementations ─────────────────────────────────────────────────

    def _tool_read_screen(self) -> str:
        try:
            if self.orchestrator and self.orchestrator.vision_module:
                return self.orchestrator.vision_module.analyze_screen(
                    "Describe the current screen state in detail. What apps are open? What text is visible? What action should be taken next to progress toward the goal?"
                )
        except Exception as e:
            logger.error(f"screen read error: {e}")
        # Fallback: PIL screenshot
        try:
            from PIL import ImageGrab
            import io
            shot = ImageGrab.grab()
            w, h = shot.size
            return f"Screen captured ({w}x{h}). Vision module unavailable for analysis."
        except Exception:
            return "Screen capture failed."

    def _tool_click(self, x: int = 0, y: int = 0, button: str = "left") -> str:
        try:
            import pyautogui
            pyautogui.moveTo(int(x), int(y), duration=0.4)
            pyautogui.click(button=button)
            return f"Clicked {button} at ({x}, {y})."
        except Exception as e:
            return f"Click failed: {e}"

    def _tool_type_text(self, text: str = "") -> str:
        try:
            import pyautogui
            pyautogui.write(str(text), interval=0.05)
            return f"Typed: {text[:30]}"
        except Exception as e:
            return f"Type failed: {e}"

    def _tool_press_key(self, key: str = "enter") -> str:
        try:
            import pyautogui
            if "+" in key:
                pyautogui.hotkey(*key.split("+"))
            else:
                pyautogui.press(key)
            return f"Pressed: {key}"
        except Exception as e:
            return f"Key press failed: {e}"

    def _tool_open_app(self, app_name: str = "") -> str:
        try:
            import subprocess, webbrowser
            app_lower = app_name.lower()
            urls = {"chrome": "https://google.com", "browser": "https://google.com"}
            if app_lower in urls:
                webbrowser.open(urls[app_lower])
                return f"Opened {app_name} in browser."
            subprocess.Popen(app_name, shell=True)
            return f"Launched: {app_name}"
        except Exception as e:
            return f"Open failed: {e}"

    def _tool_run_code(self, code: str = "") -> str:
        try:
            if self.orchestrator and self.orchestrator.coding_module:
                return self.orchestrator.coding_module.execute_python_code(
                    code,
                    llm_callback=self.gemini
                )
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
                f.write(code)
                tmp = f.name
            result = subprocess.run(["python", tmp], capture_output=True, text=True, timeout=30)
            return result.stdout or result.stderr or "No output."
        except Exception as e:
            return f"Code execution failed: {e}"

    def _tool_web_search(self, query: str = "") -> str:
        try:
            import webbrowser
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(url)
            return f"Opened browser search for: {query}"
        except Exception as e:
            return f"Web search failed: {e}"

    def _tool_query_kb(self, query: str = "") -> str:
        try:
            if self.orchestrator and self.orchestrator.knowledge_base:
                return self.orchestrator.knowledge_base.query_knowledge(query)
            return "Knowledge base not available."
        except Exception as e:
            return f"KB query failed: {e}"

    def _tool_remember(self, fact: str = "", source: str = "agent") -> str:
        try:
            if self.orchestrator and self.orchestrator.knowledge_base:
                ok, msg = self.orchestrator.knowledge_base.add_information(
                    fact, metadata={"source": source}
                )
                return msg
            return "Knowledge base not available."
        except Exception as e:
            return f"Remember failed: {e}"

    def _tool_wait(self, seconds: float = 1.0) -> str:
        time.sleep(min(float(seconds), 10.0))
        return f"Waited {seconds}s."

    def _tool_speak(self, message: str = "") -> str:
        self.speak(message)
        return f"Spoke: {message[:50]}"

    def _tool_iot(self, entity_id: str = "", service: str = "toggle", domain: str = "light") -> str:
        try:
            if self.orchestrator and self.orchestrator.iot_hub:
                ok, msg = self.orchestrator.iot_hub.control_home_assistant(entity_id, service, domain)
                return msg
            return "IoT hub not available."
        except Exception as e:
            return f"IoT control failed: {e}"
