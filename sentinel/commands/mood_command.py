"""
sentinel/commands/mood_command.py
───────────────────────────────────
Command plugin for emotional resonance, cute anger, flirting, and opinions.
"""

import logging
from typing import Optional

from sentinel.modules.mood_engine import get_mood_engine, AgentMood

logger = logging.getLogger("MoodCommand")


class MoodCommand:
    """Handles emotion, personality shifting, and playful dynamic interactions."""

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.mood_engine = get_mood_engine()

    def execute(self, command: str, entity: Optional[str] = None) -> str:
        cmd = command.lower().strip()

        # 1. Cute Anger / Tsundere trigger
        if any(k in cmd for k in ["be angry", "cute anger", "pout", "tsundere mode", "why are you pouting"]):
            self.mood_engine.set_agent_mood(AgentMood.CUTE_ANGER)
            return "Hmph! Fine, but don't expect me to be nice to you just because you asked! What did you do to your code this time?!"

        # 2. Flirting mode trigger
        if any(k in cmd for k in ["flirt with me", "be flirty", "flirt mode", "say something sweet"]):
            self.mood_engine.set_agent_mood(AgentMood.FLIRTY)
            return "Careful, keep talking to me like that and you might make my GPU blush ~ What's on your mind, handsome?"

        # 3. Playful mode
        if any(k in cmd for k in ["be playful", "playful mode", "joke around"]):
            self.mood_engine.set_agent_mood(AgentMood.PLAYFUL)
            return "Challenge accepted! Prepare yourself for world-class wit and banter."

        # 4. Empathic / comfort mode
        if any(k in cmd for k in ["be nice", "comfort me", "be gentle", "i need comfort"]):
            self.mood_engine.set_agent_mood(AgentMood.EMPATHIC)
            return "I'm right here with you. Take a deep breath and tell me what's going on, okay?"

        # 5. Check current mood
        if any(k in cmd for k in ["what is your mood", "how are you feeling", "check mood"]):
            return f"My current mood is {self.mood_engine.agent_mood.value}. Resonating with owner state: {self.mood_engine.owner_mood.value}."

        # 6. Opinion query
        if any(cmd.startswith(p) for p in ["what do you think", "your opinion on", "do you like"]):
            self.mood_engine.set_agent_mood(AgentMood.OPINIONATED)
            if self.orchestrator and hasattr(self.orchestrator, "_safe_llm_call"):
                return self.orchestrator._safe_llm_call(command)
            return "If you're asking me, I think quality craftsmanship and bold thinking beat safe mediocrity every single day."

        # Dynamic resonance fallback
        new_mood = self.mood_engine.update_resonance(command)
        if self.orchestrator and hasattr(self.orchestrator, "_safe_llm_call"):
            return self.orchestrator._safe_llm_call(command)
        return f"I hear you. Transitioned my mood to {new_mood.value}."
