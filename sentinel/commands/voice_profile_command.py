"""
sentinel/commands/voice_profile_command.py
───────────────────────────────────────────
Command plugin for recording voice samples and configuring custom agent voice profiles.
"""

import os
import logging
from typing import Optional

from sentinel.voice.voice_cloner import VoiceProfileManager

logger = logging.getLogger("VoiceProfileCommand")


class VoiceProfileCommand:
    """Handles voice cloning and reference sample recording."""

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.manager = VoiceProfileManager()

    def execute(self, command: str, entity: Optional[str] = None) -> str:
        cmd = command.lower().strip()

        # 1. Record voice sample
        if any(k in cmd for k in ["record voice sample", "record my voice", "clone my voice", "sample my voice", "record voice for agent"]):
            success, msg = self.manager.record_voice_sample(duration=5.0)
            return msg

        # 2. Reset to default voice
        if any(k in cmd for k in ["reset agent voice", "default voice", "reset voice"]):
            return self.manager.reset_to_default_voice()

        # 3. Check voice profile status
        if any(k in cmd for k in ["voice profile", "what is your voice", "show voice", "list voice"]):
            prof = self.manager.get_active_profile()
            if prof.get("is_custom"):
                return f"Active voice profile: {prof['name']} (Base: {prof['base_voice']}, Pitch: {prof['pitch_offset']}, Rate: {prof['rate_offset']})."
            return f"Active voice profile: Standard Neural ({prof['base_voice']})."

        # 4. Set from path if provided in entity
        if entity and os.path.exists(entity):
            success, msg = self.manager.set_custom_voice_from_file(entity)
            return msg

        return "Say 'record voice sample' to record a sample of your voice for the agent, or 'reset agent voice' to return to default."
