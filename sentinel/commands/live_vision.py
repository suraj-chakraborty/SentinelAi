"""
sentinel/commands/live_vision.py
───────────────────────────────
Command to expose live spatial vision capabilities.
"""

import logging
from typing import Dict, Any, Optional

from sentinel.vision.live_vision import get_live_vision, start_vision_stream, stop_vision_stream

logger = logging.getLogger("LiveVisionCommand")


class LiveVisionCommand:
    def __init__(self):
        pass

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute a vision action."""
        try:
            stream = get_live_vision()
            
            if action == "start":
                camera_id = kwargs.get("camera_id", 0)
                fps = kwargs.get("fps", 10)
                if stream.start():
                    return {"success": True, "message": f"Vision started (camera={camera_id}, fps={fps})"}
                return {"success": False, "error": "Failed to start vision"}
            
            elif action == "stop":
                stream.stop()
                return {"success": True, "message": "Vision stopped"}
            
            elif action == "capture":
                filepath = stream.capture_frame()
                if filepath:
                    return {"success": True, "filepath": filepath}
                return {"success": False, "error": "No frame available"}
            
            elif action == "analyze":
                prompt = kwargs.get("prompt", "Describe what you see.")
                result = stream.analyze_current_frame(prompt)
                return {"success": True, "analysis": result}
            
            elif action == "stats":
                return stream.get_stats()
            
            elif action == "frame":
                frame = stream.get_frame_base64()
                if frame:
                    return {"success": True, "frame": frame}
                return {"success": False, "error": "No frame available"}
            
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        
        except Exception as e:
            logger.error(f"Vision command failed: {e}")
            return {"success": False, "error": str(e)}


_live_vision_command: Optional[LiveVisionCommand] = None


def get_live_vision_command() -> LiveVisionCommand:
    global _live_vision_command
    if _live_vision_command is None:
        _live_vision_command = LiveVisionCommand()
    return _live_vision_command