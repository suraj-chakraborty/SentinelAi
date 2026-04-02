"""
sentinel/core/swarm_engine.py
─────────────────────────────
Phase 5: Distributed Swarm Intelligence.
Allows Sentinel to delegate parts of a task to specialized Sub-Agents that
communicate via JSON messaging.
"""

import json
import logging
from typing import List, Dict

logger = logging.getLogger("SwarmEngine")

class SwarmAgent:
    def __init__(self, role: str, instruction: str, llm_callback):
        self.role = role
        self.instruction = instruction
        self.llm_callback = llm_callback

    def execute(self, task: str, shared_memory: list) -> str:
        context = "\n".join([f"({msg['role']}): {msg['content']}" for msg in shared_memory[-3:]])
        prompt = (
            f"You are a specialized Swarm Agent. Your Role: {self.role}.\n"
            f"Instructions: {self.instruction}\n\n"
            f"Recent Swarm Context:\n{context}\n\n"
            f"Your Task: {task}\n"
            f"Execute your task and return the result."
        )
        try:
            return self.llm_callback(prompt)
        except Exception as e:
            return f"Agent {self.role} failed: {e}"


class SwarmOrchestrator:
    """Manages the lifecycle of multi-agent tasks."""
    def __init__(self, llm_callback):
        self.llm_callback = llm_callback
        
        # Instantiate built-in experts
        self.agents = {
            "researcher": SwarmAgent(
                "Researcher", 
                "You find information, summarize facts, and provide raw data.", 
                llm_callback
            ),
            "coder": SwarmAgent(
                "Senior Developer", 
                "You write clear, bug-free, optimal code based on research provided.", 
                llm_callback
            ),
            "formatter": SwarmAgent(
                "Editor", 
                "You format output into beautiful Markdown with headings and bold text.", 
                llm_callback
            )
        }

    def run_swarm(self, goal: str, agents_needed: List[str] = None) -> str:
        logger.info(f"Swarm activated for goal: {goal}")
        
        if not agents_needed:
            agents_needed = ["researcher", "coder", "formatter"]
            
        shared_memory = [{"role": "user", "content": goal}]
        
        for agent_name in agents_needed:
            if agent_name not in self.agents:
                continue
                
            agent = self.agents[agent_name]
            logger.info(f"Swarm -> delegating to {agent.role}")
            
            result = agent.execute(goal, shared_memory)
            shared_memory.append({"role": agent.role, "content": result})
            
        return shared_memory[-1]["content"]
