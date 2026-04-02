"""
sentinel/modules/robotics_module.py
───────────────────────────────────
Sci-Fi Tier 1: Embodied AI (Robotics).
Sends abstract JSON commands via UDP broadcast to any listening node 
(e.g., Raspberry Pi rover, smart switch, or experimental ROS robot) on the local WiFi.
"""

import json
import socket
import logging

logger = logging.getLogger("RoboticsModule")

class RoboticsModule:
    def __init__(self, port: int = 4242):
        self.port = port
        self.broadcast_ip = "255.255.255.255"
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            logger.info(f"Robotics module bound to UDP broadcast port {self.port}")
        except Exception as e:
            logger.error(f"Failed to bind UDP socket: {e}")
            self.sock = None

    def instruct_automata(self, prompt: str, llm_callback) -> str:
        """Uses LLM to convert a spoken command into raw JSON kinematics/actions and broadcasts it."""
        if not self.sock:
            return "Robotics interface is offline."

        sys_prompt = (
            "You are a robotic control translator. Convert the following natural language command into a strict "
            "JSON payload dictating robotic action. Return ONLY the JSON object.\n"
            "Format:\n"
            '{"action": "move|turn|grab|stop", "vector": [x, y, z], "speed": 1.0, "duration_sec": 2}\n'
            f"Command: {prompt}"
        )

        try:
            response = llm_callback(sys_prompt)
            # Extrac JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
                
            payload = json.loads(response)
            encoded = json.dumps(payload).encode('utf-8')
            
            # Broadcast to entire subnet
            self.sock.sendto(encoded, (self.broadcast_ip, self.port))
            logger.info(f"Broadcasted robotic command: {payload}")
            return f"Transmitted robotic command: {payload.get('action')} at speed {payload.get('speed', 'auto')}."
        except Exception as e:
            logger.error(f"Robotics error: {e}")
            return f"Failed to translate command for automata. ({e})"
