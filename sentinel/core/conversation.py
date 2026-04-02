"""
sentinel/core/conversation.py
──────────────────────────────
Multi-turn conversation context manager.

Maintains a rolling window of user/assistant exchanges and injects
them into LLM prompts so SentinelAI remembers what was said earlier
in the session.
"""

import time
import json
import os
import logging
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Optional

logger = logging.getLogger("SentinelConversation")

APPDATA_DIR = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi")
HISTORY_PATH = os.path.join(APPDATA_DIR, "conversation_history.json")


@dataclass
class Turn:
    role: str           # "user" or "assistant"
    content: str
    timestamp: float
    emotion: Optional[str] = None
    intent: Optional[str] = None


class ConversationManager:
    """
    Manages multi-turn conversation history for coherent AI interactions.
    
    Features:
    - Rolling window (configurable max turns)
    - Persistent storage across sessions
    - Emotion + intent metadata per turn
    - Context-aware LLM prompt builder
    """

    SYSTEM_PROMPT = """You are SentinelAI — an advanced, personal AI assistant running locally on the user's Windows PC.

Your personality:
- Concise and action-oriented (prefer doing over explaining)
- Proactively helpful (suggest next steps when relevant)  
- Privacy-first (always remind the user their data stays local)
- Slightly formal but warm — like a highly capable executive assistant

Capabilities you have:
- Voice control, wake-word detection, voice biometrics
- Full desktop automation (open apps, window management, GUI control)
- Screen vision (analyze what's on screen)
- IoT control (Home Assistant, MQTT)
- Personal knowledge base (RAG memory)
- Encrypted secrets vault
- Local code generation and execution
- Calendar & meeting management
- File intelligence (summarize PDFs, DOCX)
- Gesture control via webcam
- Security monitoring (USB, processes, network)
- Habit learning and proactive suggestions
- Web automation (Playwright, Selenium)
- Daily briefing generation

When you can ACTION something directly, do it and confirm briefly.
When you need to just respond, keep it under 2 sentences unless asked for detail.
Always speak in first person as SentinelAI."""

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._history: deque = deque(maxlen=max_turns)
        self._session_start = time.time()
        self._load_recent_history()

    def add_turn(self, role: str, content: str, emotion: str = None, intent: str = None):
        """Record a conversation turn."""
        turn = Turn(
            role=role,
            content=content[:2000],  # Cap to avoid huge prompts
            timestamp=time.time(),
            emotion=emotion,
            intent=intent
        )
        self._history.append(turn)
        self._save_history()

    def build_messages(self, user_message: str, emotion: str = None) -> List[dict]:
        """
        Build the full message list for an LLM API call.
        Includes system prompt + history + current user message.
        """
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Add conversation history
        for turn in list(self._history):
            messages.append({
                "role": turn.role,
                "content": turn.content
            })

        # Add current user message (with emotion context if detected)
        user_content = user_message
        if emotion and emotion != "Neutral":
            user_content = f"[User sounds {emotion.lower()}] {user_message}"

        messages.append({"role": "user", "content": user_content})
        return messages

    def get_recent_context(self, n: int = 5) -> str:
        """Returns a plain-text summary of the last N turns for logging/display."""
        turns = list(self._history)[-n:]
        lines = []
        for t in turns:
            ts = time.strftime("%H:%M", time.localtime(t.timestamp))
            lines.append(f"[{ts}] {t.role.upper()}: {t.content[:100]}")
        return "\n".join(lines)

    def clear_session(self):
        """Clear current session history."""
        self._history.clear()
        logger.info("Conversation history cleared.")

    def _save_history(self):
        """Persist recent history to disk."""
        try:
            os.makedirs(APPDATA_DIR, exist_ok=True)
            # Only save last 10 turns for next session
            recent = [asdict(t) for t in list(self._history)[-10:]]
            with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(recent, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")

    def _load_recent_history(self):
        """Load recent history from the last session."""
        try:
            if os.path.exists(HISTORY_PATH):
                with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Only load turns from last 2 hours
                cutoff = time.time() - 7200
                for t in data:
                    if t.get("timestamp", 0) > cutoff:
                        self._history.append(Turn(**t))
                if self._history:
                    logger.info(f"Loaded {len(self._history)} turns from previous session.")
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")


# Module-level singleton
_conversation_manager: ConversationManager = None


def get_conversation() -> ConversationManager:
    """Get or create the global conversation manager."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager
