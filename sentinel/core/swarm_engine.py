"""
sentinel/core/swarm_engine.py
─────────────────────────────
Advanced Swarm Intelligence.
Implements Planner-Worker architecture for multi-agent tasks.
"""

import json
import logging
import threading
from typing import List, Dict, Optional

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

    def run_swarm(self, goal: str) -> str:
        """
        Main entry point for multi-agent delegation.
        Flow: Planner -> Parallel Workers -> Editor.
        """
        logger.info(f"Swarm active for goal: {goal}")
        
        # 1. Planner Phase
        plan = self._generate_plan(goal)
        if not plan:
            return "I couldn't develop a multi-agent plan for this goal."
            
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
        final_output = self.agents["editor"].execute(
            f"Synthesize the final response for: {goal}", 
            shared_memory
        )
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
        output_map[agent_id] = self.agents[agent_id].execute(task, memory)
