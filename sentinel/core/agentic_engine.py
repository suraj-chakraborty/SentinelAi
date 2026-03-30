import json
import logging
import time

class AgenticEngine:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("AgenticEngine")
        self.max_steps = 5

    def execute_autonomous_goal(self, goal):
        """Attempts to achieve a complex goal by iterating through vision, planning, and action."""
        self.logger.info(f"Starting agentic loop for goal: {goal}")
        
        current_step = 0
        history = []
        
        while current_step < self.max_steps:
            # 1. PERCEPTION: See the screen
            screen_context = self.orchestrator.vision_module.analyze_screen(
                f"I am trying to achieve the goal: {goal}. What is currently visible on the screen that is relevant?"
            )
            
            # 2. PLANNING: Decide next action
            plan_prompt = f"""
            Goal: {goal}
            Current Step: {current_step + 1}/{self.max_steps}
            History: {history}
            Screen Context: {screen_context}
            
            Based on the above, what is the single next best action to take? 
            Return a JSON object with:
            - "thought": Reasoning for this action
            - "action_type": One of [COMMAND, CLICK, TYPE, WAIT, DONE]
            - "payload": The specific command string, (x,y) coordinates, or text to type.
            """
            
            plan_response = self.orchestrator._safe_llm_call(plan_prompt)
            try:
                # Clean and parse JSON
                if "```json" in plan_response:
                    plan_response = plan_response.split("```json")[1].split("```")[0].strip()
                action = json.loads(plan_response)
                
                if action['action_type'] == 'DONE':
                    return f"Goal achieved: {action['thought']}"
                
                # 3. EXECUTION: Perform the action
                result = self._perform_action(action)
                history.append({"action": action, "result": result})
                
            except Exception as e:
                self.logger.error(f"Agentic step failed: {e}")
                history.append({"error": str(e)})
            
            current_step += 1
            time.sleep(2) # Brief pause between steps
            
        return "Task timed out before goal was achieved. History: " + str(history)

    def _perform_action(self, action):
        atype = action['action_type']
        payload = action['payload']
        
        if atype == 'COMMAND':
            return self.orchestrator._execute_single_command(payload)
        elif atype == 'CLICK':
            # Payload should be (x,y)
            return self.orchestrator.automation_module.move_and_click(payload[0], payload[1])
        elif atype == 'TYPE':
            return self.orchestrator.automation_module.type_text(payload)
        elif atype == 'WAIT':
            time.sleep(float(payload))
            return "Waited."
        return "Unknown action type."
