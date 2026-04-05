"""
sentinel/core/semantic_intent.py
──────────────────────────────────
Sentence-embedding fallback classifier for intent detection.

Used only when keyword + fuzzy routing return UNKNOWN.
Lazy-loads all-MiniLM-L6-v2 on first use (thread-safe).

v2 — Extended prototype utterances to cover all 35 intents in the taxonomy.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("SemanticIntent")

_DEFAULT_MIN_SIM = 0.36


def _semantic_disabled() -> bool:
    v = (os.getenv("SENTINEL_SEMANTIC_ROUTING") or "").strip().lower()
    return v in ("0", "false", "no", "off")


# ── Prototype utterances per intent ──────────────────────────────────────────
# Intent key must match Intent.value string in intent_detector.py

PROTOTYPE_UTTERANCES: Dict[str, List[str]] = {
    "open_app": [
        "open chrome", "launch notepad", "start the calculator",
        "run vscode", "open my browser", "start microsoft edge",
        "open spotify", "launch discord", "open file explorer",
    ],
    "close_app": [
        "close chrome", "exit notepad", "quit discord",
        "stop spotify", "close the browser", "kill firefox",
        "terminate calculator", "close this application",
    ],
    "browse_url": [
        "go to google.com", "navigate to github", "open youtube website",
        "browse to amazon", "visit wikipedia.org",
    ],
    "shutdown": [
        "shut down the computer", "turn off my pc", "power off the machine",
        "shutdown windows", "switch off computer",
    ],
    "restart": [
        "restart the computer", "reboot windows", "restart my pc",
        "reboot the machine", "restart everything",
    ],
    "lock_pc": [
        "lock my computer", "lock the screen", "secure my pc",
    ],
    "system_status": [
        "how much memory is left", "cpu usage", "check ram",
        "system temperature", "is my disk full", "battery status",
        "computer performance", "how much storage do i have",
    ],
    "volume_up": [
        "turn up the volume", "make it louder", "increase sound",
        "volume higher",
    ],
    "volume_down": [
        "turn down the volume", "make it quieter", "lower the sound",
        "decrease volume",
    ],
    "volume_mute": [
        "mute the sound", "unmute audio", "silence the speakers",
        "toggle mute",
    ],
    "take_screenshot": [
        "take a screenshot", "capture my screen", "screenshot please",
        "grab a screenshot",
    ],
    "play_music": [
        "play some music", "start playing music", "play a song",
        "resume music", "play my playlist",
    ],
    "pause_music": [
        "pause the music", "stop playing", "pause song",
        "stop the music",
    ],
    "skip_track": [
        "skip this song", "next track please", "play the next song",
        "previous track", "go back one song",
    ],
    "set_timer": [
        "set a timer for 5 minutes", "start a 30 second timer",
        "remind me in 10 minutes with a timer", "countdown timer please",
    ],
    "set_alarm": [
        "set an alarm for 7am", "wake me up at 6 thirty",
        "alarm at eight pm", "set morning alarm",
    ],
    "set_reminder": [
        "remind me to take my medicine at noon", "set a reminder for meeting",
        "remind me in 2 hours", "dont let me forget the call",
    ],
    "calendar_read": [
        "what's on my calendar today", "show my schedule", "my agenda",
        "upcoming meetings", "what do i have this week",
    ],
    "calendar_add": [
        "add meeting to my calendar", "schedule appointment friday",
        "book a slot at 3pm", "add event to google calendar",
    ],
    "search_web": [
        "search for latest ai news", "look up nearest coffee shop",
        "google how to cook pasta", "find out about quantum computing",
        "tell me about black holes",
    ],
    "get_weather": [
        "what's the weather like today", "will it rain tomorrow",
        "temperature outside", "weather forecast for this week",
        "is it sunny",
    ],
    "get_news": [
        "what are the headlines", "top news today", "latest news",
        "what's happening in the world", "show me the news",
    ],
    "calculate": [
        "what is 15 percent of 200", "calculate 44 times 17",
        "how much is 120 divided by 8", "compute square root of 144",
    ],
    "translate": [
        "translate hello to spanish", "how do you say thank you in french",
        "what is goodbye in japanese", "translate this to german",
    ],
    "send_email": [
        "send an email to john", "compose email to boss",
        "email the team about the meeting", "write email to client",
    ],
    "read_email": [
        "check my email", "read my inbox", "any new emails",
        "read my latest messages", "show unread emails",
    ],
    "find_file": [
        "find the report file", "where is my document", "search for pdf files",
        "locate the spreadsheet", "find the photo i took yesterday",
    ],
    "read_file": [
        "read the file called notes.txt", "open and read the document",
        "what's in that text file", "show me the contents of the config file",
    ],
    "write_file": [
        "create a new file called todo.txt", "write my notes to a file",
        "save this information to a file",
    ],
    "delete_file": [
        "delete the old report", "remove that file", "trash the document",
        "permanently delete logs",
    ],
    "take_note": [
        "take a note: buy milk", "jot this down", "write down: call dentist",
        "add a note that i need to review the report",
    ],
    "read_note": [
        "what are my notes", "show me my notes", "read back my notes",
        "list all notes",
    ],
    "kb_query": [
        "what do you know about my project", "recall what i told you about",
        "do you remember anything about", "search your memory for",
    ],
    "kb_add": [
        "remember that i prefer dark mode", "save this to your memory",
        "learn this fact about me", "add to your knowledge base",
    ],
    "clipboard_read": [
        "what did i copy", "read my clipboard", "show clipboard content",
        "what's in my clipboard",
    ],
    "clipboard_write": [
        "copy this text to clipboard", "save to clipboard", "add to my clipboard",
    ],
    "code_execute": [
        "run this python script", "execute the code", "run code to rename files",
        "write and run a script that",
    ],
    "computer_use": [
        "click the submit button on screen", "type into the search box",
        "control my desktop to fill in the form", "use the computer to open settings",
        "do this on screen for me",
    ],
    "autonomous_agent": [
        "organize my downloads folder", "find all pdf files on desktop",
        "automate filling this form", "summarize this long page",
        "create a folder structure for my project",
    ],
    "swarm_task": [
        "research and write a report on", "send multiple agents to investigate",
        "use a team of ai agents to", "swarm research this topic",
    ],
    "deactivate": [
        "exit sentinel", "quit the assistant", "close sentinel ai",
        "deactivate the ai", "stop listening sentinel", "goodbye sentinel",
    ],
}


class SemanticIntentClassifier:
    """
    Lazy-loads MiniLM-L6-v2 and compares utterance to per-intent prototype
    cluster centroids (mean-pooled, L2-normalised).
    """

    def __init__(self, min_similarity: float = _DEFAULT_MIN_SIM):
        self.min_similarity = min_similarity
        self._lock = threading.Lock()
        self._model = None
        self._intent_ids: List[str] = []
        self._matrix = None   # (n_intents, dim) row-normalised

    def _ensure_loaded(self) -> bool:
        if _semantic_disabled():
            return False
        with self._lock:
            if self._matrix is not None:
                return True
            try:
                import numpy as np
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.info(
                    "Semantic routing disabled: install sentence-transformers "
                    "(pip install sentence-transformers)."
                )
                return False

            try:
                model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as exc:
                logger.warning("Could not load embedding model: %s", exc)
                return False

            intent_ids: List[str] = []
            vectors: List = []

            for key, phrases in PROTOTYPE_UTTERANCES.items():
                if not phrases:
                    continue
                emb = model.encode(phrases, convert_to_numpy=True, show_progress_bar=False)
                mean = emb.mean(axis=0)
                norm = np.linalg.norm(mean)
                if norm > 0:
                    mean = mean / norm
                intent_ids.append(key)
                vectors.append(mean)

            if not vectors:
                return False

            self._model = model
            self._intent_ids = intent_ids
            self._matrix = np.stack(vectors, axis=0)
            logger.info("Semantic intent routing enabled (%d labels).", len(intent_ids))
            return True

    def is_available(self) -> bool:
        return self._ensure_loaded()

    def predict(self, command: str) -> Tuple[Optional[str], float]:
        """
        Returns (intent_value_str, similarity_score) or (None, 0.0) when
        nothing exceeds the minimum similarity threshold.
        """
        text = (command or "").strip()
        if not text or len(text) > 512:
            return None, 0.0

        if not self._ensure_loaded():
            return None, 0.0

        import numpy as np

        q = self._model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        nq = np.linalg.norm(q)
        if nq > 0:
            q = q / nq

        sims = self._matrix @ q
        idx = int(np.argmax(sims))
        score = float(sims[idx])

        if score < self.min_similarity:
            return None, score
        return self._intent_ids[idx], score


# ── Singleton ─────────────────────────────────────────────────────────────────

_default_classifier: Optional[SemanticIntentClassifier] = None


def get_semantic_classifier() -> SemanticIntentClassifier:
    global _default_classifier
    if _default_classifier is None:
        raw = (os.getenv("SENTINEL_SEMANTIC_MIN_SIM") or "").strip()
        min_sim = _DEFAULT_MIN_SIM
        if raw:
            try:
                min_sim = float(raw)
            except ValueError:
                pass
        _default_classifier = SemanticIntentClassifier(min_similarity=min_sim)
    return _default_classifier
