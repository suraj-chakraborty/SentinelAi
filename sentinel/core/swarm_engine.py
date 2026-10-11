"""
sentinel/core/swarm_engine.py
─────────────────────────────
Advanced Swarm Intelligence.
Implements Planner-Worker architecture for multi-agent tasks.
"""

import json
import logging
import threading
from typing import List, Dict, Optional, Any

logger = logging.getLogger("SwarmEngine")

class SwarmAgent:
    def __init__(self, role: str, instruction: str, llm_callback):
        self.role = role
        self.instruction = instruction
        self.llm_callback = llm_callback

    def execute(self, task: str, shared_memory: list) -> str:
        context = "\n".join([f"({msg['role']}): {msg['content'][:200]}..." for msg in shared_memory[-5:]])
        prompt = (
            f"You are a specialized Swarm Agent. Role: {self.role}.\n"
            f"Instructions: {self.instruction}\n\n"
            f"Context from other agents:\n{context}\n\n"
            f"Your Specific Task: {task}\n"
            "Provide the detailed output for your part of the swarm goal."
        )
        try:
            return self.llm_callback(prompt)
        except Exception as e:
            logger.error(f"Agent {self.role} error: {e}")
            return f"Error: {e}"

class SwarmOrchestrator:
    def __init__(self, llm_callback):
        self.llm = llm_callback
        self.agents = {
            "researcher": SwarmAgent("Researcher", "Find and summarize facts using search results.", llm_callback),
            "coder": SwarmAgent("Senior Developer", "Write high-quality Python code and documentation.", llm_callback),
            "editor": SwarmAgent("Editor", "Format, polish, and synthesize multiple agent outputs into a final report.", llm_callback),
            "security_analyst": SwarmAgent("Security Analyst", "Review code and plans for vulnerabilities.", llm_callback)
        }
        self.current_goal: str = ""
        self.status: str = "idle"
        self.plan: Dict[str, str] = {}
        self.agent_statuses: Dict[str, Dict[str, Any]] = {
            name: {"role": agent.role, "status": "idle", "task": "", "output": ""}
            for name, agent in self.agents.items()
        }
        self.history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def get_state(self) -> Dict[str, Any]:
        """Returns the current multi-agent swarm state and task DAG for HUD visualization."""
        with self._lock:
            return {
                "goal": self.current_goal,
                "status": self.status,
                "plan": dict(self.plan),
                "agents": {k: dict(v) for k, v in self.agent_statuses.items()},
                "history_count": len(self.history),
                "active_count": sum(1 for a in self.agent_statuses.values() if a["status"] == "running")
            }

    def run_swarm(self, goal: str) -> str:
        """
        Main entry point for multi-agent delegation.
        Flow: Planner -> Parallel Workers -> Editor.
        """
        logger.info(f"Swarm active for goal: {goal}")
        with self._lock:
            self.current_goal = goal
            self.status = "planning"
            for k in self.agent_statuses:
                self.agent_statuses[k]["status"] = "idle"
                self.agent_statuses[k]["task"] = ""
                self.agent_statuses[k]["output"] = ""

        # 1. Planner Phase
        plan = self._generate_plan(goal)
        if not plan:
            with self._lock:
                self.status = "error"
            return "I couldn't develop a multi-agent plan for this goal."

        with self._lock:
            self.plan = plan
            self.status = "executing"
            for agent_id, sub_task in plan.items():
                if agent_id in self.agent_statuses:
                    self.agent_statuses[agent_id]["task"] = sub_task
                    self.agent_statuses[agent_id]["status"] = "running"

        shared_memory = [{"role": "user", "content": goal}]
        results = {}

        # 2. Worker Phase (Parallel)
        threads = []
        for agent_id, sub_task in plan.items():
            if agent_id in self.agents and agent_id != "editor":
                t = threading.Thread(
                    target=self._run_agent_thread, 
                    args=(agent_id, sub_task, shared_memory, results)
                )
                threads.append(t)
                t.start()

        for t in threads:
            t.join()

        # Update shared memory with worker results
        for agent_id, output in results.items():
            shared_memory.append({"role": f"Agent_{agent_id}", "content": output})

        # 3. Final Synthesis (Editor)
        logger.info("Swarm -> Final synthesis by Editor.")
        with self._lock:
            self.status = "synthesizing"
            if "editor" in self.agent_statuses:
                self.agent_statuses["editor"]["status"] = "running"
                self.agent_statuses["editor"]["task"] = "Synthesizing worker outputs into unified response"

        final_output = self.agents["editor"].execute(
            f"Synthesize the final response for: {goal}", 
            shared_memory
        )

        with self._lock:
            if "editor" in self.agent_statuses:
                self.agent_statuses["editor"]["status"] = "done"
                self.agent_statuses["editor"]["output"] = final_output[:300]
            self.status = "completed"
            self.history.append({
                "goal": goal,
                "plan": plan,
                "output_preview": final_output[:200]
            })

        return final_output

    def _generate_plan(self, goal: str) -> Dict[str, str]:
        """Ask the LLM to break the goals into sub-tasks for specific agents."""
        logger.info("Swarm -> Generating plan...")
        prompt = f"""
        Goal: {goal}
        
        Available agents:
        - researcher: facts, data gathering
        - coder: script writing, algorithm design
        - security_analyst: auditing, risk check
        
        Task: Break this goal into 2-3 specific sub-tasks.
        Respond with ONLY a JSON map of agent name to sub-task.
        Example: {{"researcher": "Find current stock price of AAPL", "coder": "Create a plot script"}}
        """
        try:
            resp = self.llm(prompt)
            import re
            match = re.search(r'\{[^{}]+\}', resp, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.error(f"Planner error: {e}")
        return {}

    def _run_agent_thread(self, agent_id: str, task: str, memory: list, output_map: dict):
        logger.info(f"Swarm -> Starting {agent_id}...")
        res = self.agents[agent_id].execute(task, memory)
        output_map[agent_id] = res
        with self._lock:
            if agent_id in self.agent_statuses:
                self.agent_statuses[agent_id]["status"] = "done"
                self.agent_statuses[agent_id]["output"] = str(res)[:300]


_global_swarm: Optional[SwarmOrchestrator] = None
_swarm_lock = threading.Lock()


def get_swarm_orchestrator(llm_callback=None) -> SwarmOrchestrator:
    global _global_swarm
    with _swarm_lock:
        if _global_swarm is None:
            if llm_callback is None:
                from sentinel.app.intelligence import ask_gemini
                llm_callback = ask_gemini
            _global_swarm = SwarmOrchestrator(llm_callback)
        elif llm_callback is not None:
            _global_swarm.llm = llm_callback
        return _global_swarm


def get_swarm_state() -> Dict[str, Any]:
    try:
        return get_swarm_orchestrator().get_state()
    except Exception as exc:
        return {"status": "error", "error": str(exc), "agents": {}}

