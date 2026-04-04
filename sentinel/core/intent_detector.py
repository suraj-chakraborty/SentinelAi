import logging
import re
from typing import Optional

try:
    from rapidfuzz import process, fuzz
    _HAS_FUZZY = True
except ImportError:
    _HAS_FUZZY = False

logger = logging.getLogger("IntentDetector")

from enum import Enum

class Intent(Enum):
    OPEN_APP = "open_app"
    CLOSE_APP = "close_app"
    SHUTDOWN = "shutdown"
    RESTART = "restart"
    SYSTEM_STATUS = "system_status"
    AUTONOMOUS_AGENT = "autonomous_agent"
    DEACTIVATE = "deactivate"
    UNKNOWN = "unknown"

# Define base intents with their keywords
INTENTS = {
    Intent.OPEN_APP: ["open", "launch", "start", "run", "execute"],
    Intent.CLOSE_APP: ["close", "exit", "terminate", "stop", "kill"],
    Intent.SHUTDOWN: ["shutdown", "turn off computer", "power off"],
    Intent.RESTART: ["restart", "reboot"],
    Intent.SYSTEM_STATUS: ["system status", "system info", "cpu", "ram", "temp", "status"],
    Intent.AUTONOMOUS_AGENT: [
        "achieve", "goal", "build", "organize", "search for", "find all", "find my",
        "how to", "analyze", "analyse", "on my screen", "automate", "install ",
    ],
    Intent.DEACTIVATE: ["destroy yourself", "self destruct", "self-destruct", "deactivate", "close yourself", "quit app", "exit sentinel"],
}

# Labels returned by semantic_intent.predict → Intent (string keys match Intent.value)
_SEMANTIC_LABEL_TO_INTENT = {
    "open_app": Intent.OPEN_APP,
    "close_app": Intent.CLOSE_APP,
    "shutdown": Intent.SHUTDOWN,
    "restart": Intent.RESTART,
    "system_status": Intent.SYSTEM_STATUS,
    "autonomous_agent": Intent.AUTONOMOUS_AGENT,
    "deactivate": Intent.DEACTIVATE,
}


class IntentDetector:
    """
    Hybrid routing: regex keywords (priority-ordered) → fuzzy keywords → optional embeddings.
    """

    def __init__(
        self,
        threshold: int = 70,
        semantic_enabled: Optional[bool] = None,
    ):
        self.threshold = threshold
        # None = respect env + lazy load inside semantic layer; False = never embed (tests)
        self.semantic_enabled = semantic_enabled

    def _detect_keywords(self, command: str) -> Intent:
        priority_order = [
            Intent.DEACTIVATE,
            Intent.SYSTEM_STATUS,
            Intent.AUTONOMOUS_AGENT,
            Intent.CLOSE_APP,
            Intent.OPEN_APP,
            Intent.RESTART,
            Intent.SHUTDOWN,
        ]
        for intent in priority_order:
            keywords = INTENTS.get(intent, [])
            for k in keywords:
                if re.search(r"\b" + re.escape(k) + r"\b", command):
                    return intent
        return Intent.UNKNOWN

    def _detect_fuzzy(self, command: str) -> Intent:
        if not _HAS_FUZZY:
            return Intent.UNKNOWN
        best_score = 0
        best_intent = Intent.UNKNOWN
        for intent, keywords in INTENTS.items():
            _, score, _ = process.extractOne(command, keywords, scorer=fuzz.partial_ratio)
            if score > best_score:
                best_score = score
                best_intent = intent
        if best_score >= self.threshold:
            return best_intent
        return Intent.UNKNOWN

    def _detect_semantic(self, command: str) -> Intent:
        if self.semantic_enabled is False:
            return Intent.UNKNOWN
        try:
            from sentinel.core.semantic_intent import get_semantic_classifier
        except ImportError:
            return Intent.UNKNOWN
        clf = get_semantic_classifier()
        label, score = clf.predict(command)
        if not label:
            return Intent.UNKNOWN
        intent = _SEMANTIC_LABEL_TO_INTENT.get(label)
        if intent is not None:
            logger.debug("Semantic intent: %s (score=%.3f)", label, score)
            return intent
        return Intent.UNKNOWN

    def detect_intent(self, command: str) -> Intent:
        """
        Detects intent: keywords first, then fuzzy match, then sentence-embedding similarity
        when still unknown (lazy model load).
        """
        if not command:
            return Intent.UNKNOWN

        hit = self._detect_keywords(command)
        if hit != Intent.UNKNOWN:
            return hit

        hit = self._detect_fuzzy(command)
        if hit != Intent.UNKNOWN:
            return hit

        hit = self._detect_semantic(command)
        if hit != Intent.UNKNOWN:
            return hit

        return Intent.UNKNOWN

    def extract_entity(self, command: str, intent: Intent) -> str:
        """
        Extracts the primary entity (e.g., app name) from the command based on intent.
        """
        if intent == Intent.OPEN_APP:
            for k in INTENTS[Intent.OPEN_APP]:
                if k in command:
                    parts = command.split(k, 1)
                    if len(parts) > 1:
                        entity = parts[1].strip()
                        # Clean up conjunctions, search terms, and fillers
                        splitters = [" and ", " then ", " search for ", " for ", " the ", " please ", " app ", " machine ", " in my pc", " on my pc", " in my computer", " on my screen"]
                        for splitter in splitters:
                            if splitter in entity:
                                entity = entity.split(splitter, 1)[0].strip()
                        
                        # Remove trailing sentence markers from STT
                        entity = entity.strip(".!?,")
                        return entity
        
        elif intent == Intent.CLOSE_APP:
            for k in INTENTS[Intent.CLOSE_APP]:
                if k in command:
                    parts = command.split(k, 1)
                    if len(parts) > 1:
                        return parts[1].strip()
                        
        elif intent == Intent.AUTONOMOUS_AGENT:
            # For agent, the entity is the whole goal
            return command 

        return ""
