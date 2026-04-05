"""
sentinel/core/intent_detector.py
──────────────────────────────────
Hybrid intent detection: keyword → fuzzy → semantic (MiniLM).

v2 changes
──────────
  • Expanded from 7 to 35 intents covering the full personal-assistant domain.
  • Added slot filling: `extract_slots()` returns a dict of named entities
    from a single command (e.g. app, url, query, duration, volume_level).
  • Confidence scoring passed through all three tiers.
  • Semantic routing only enabled when sentence-transformers is installed;
    the system degrades gracefully to keyword/fuzzy if it is not.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("IntentDetector")


# ── Intent Taxonomy ───────────────────────────────────────────────────────────

class Intent(Enum):
    # ── Application control ──────────────────────────────────────────────────
    OPEN_APP         = "open_app"
    CLOSE_APP        = "close_app"
    BROWSE_URL       = "browse_url"

    # ── System ───────────────────────────────────────────────────────────────
    SHUTDOWN         = "shutdown"
    RESTART          = "restart"
    LOCK_PC          = "lock_pc"
    SYSTEM_STATUS    = "system_status"
    VOLUME_UP        = "volume_up"
    VOLUME_DOWN      = "volume_down"
    VOLUME_MUTE      = "volume_mute"
    TAKE_SCREENSHOT  = "take_screenshot"

    # ── Media / Music ─────────────────────────────────────────────────────────
    PLAY_MUSIC       = "play_music"
    PAUSE_MUSIC      = "pause_music"
    SKIP_TRACK       = "skip_track"

    # ── Time management ───────────────────────────────────────────────────────
    SET_TIMER        = "set_timer"
    SET_ALARM        = "set_alarm"
    SET_REMINDER     = "set_reminder"
    CALENDAR_READ    = "calendar_read"
    CALENDAR_ADD     = "calendar_add"

    # ── Web / Search ──────────────────────────────────────────────────────────
    SEARCH_WEB       = "search_web"
    GET_WEATHER      = "get_weather"
    GET_NEWS         = "get_news"
    CALCULATE        = "calculate"
    TRANSLATE        = "translate"

    # ── Communication ─────────────────────────────────────────────────────────
    SEND_EMAIL       = "send_email"
    READ_EMAIL       = "read_email"

    # ── Files ─────────────────────────────────────────────────────────────────
    FIND_FILE        = "find_file"
    READ_FILE        = "read_file"
    WRITE_FILE       = "write_file"
    DELETE_FILE      = "delete_file"

    # ── Notes / Knowledge ─────────────────────────────────────────────────────
    TAKE_NOTE        = "take_note"
    READ_NOTE        = "read_note"
    KB_QUERY         = "kb_query"
    KB_ADD           = "kb_add"

    # ── Clipboard ─────────────────────────────────────────────────────────────
    CLIPBOARD_READ   = "clipboard_read"
    CLIPBOARD_WRITE  = "clipboard_write"

    # ── Code execution ────────────────────────────────────────────────────────
    CODE_EXECUTE     = "code_execute"

    # ── Computer use / Desktop automation ────────────────────────────────────
    COMPUTER_USE     = "computer_use"

    # ── Agents ────────────────────────────────────────────────────────────────
    AUTONOMOUS_AGENT = "autonomous_agent"
    SWARM_TASK       = "swarm_task"

    # ── Session ───────────────────────────────────────────────────────────────
    DEACTIVATE       = "deactivate"
    DESCRIBE_SCREEN  = "describe_screen"
    UNKNOWN          = "unknown"


# ── Keyword Rules ─────────────────────────────────────────────────────────────
# Each rule: (priority, list-of-trigger-phrases)
# Priority: lower = checked first.  All phrases checked as substrings.

_KEYWORD_RULES: List[Tuple[int, Intent, List[str]]] = [
    # Session
    (0, Intent.DEACTIVATE,       ["exit sentinel", "quit sentinel", "deactivate", "goodbye sentinel", "shut yourself down", "destroy yourself"]),
    # System destructive — high priority
    (1, Intent.SHUTDOWN,         ["shutdown computer", "shut down computer", "turn off computer", "power off", "shutdown pc", "shut down pc"]),
    (1, Intent.RESTART,          ["restart computer", "reboot computer", "restart pc", "reboot pc", "restart windows", "reboot"]),
    (1, Intent.LOCK_PC,          ["lock computer", "lock pc", "lock screen", "lock the screen"]),
    # App control
    (2, Intent.OPEN_APP,         ["open ", "launch ", "start ", "run "]),
    (2, Intent.CLOSE_APP,        ["close ", "exit ", "quit ", "kill ", "terminate "]),
    (2, Intent.BROWSE_URL,       ["go to ", "navigate to ", "open website", "browse to ", "visit "]),
    # Perception
    (4, Intent.TAKE_SCREENSHOT,  ["screenshot", "take a screenshot", "capture screen", "screen capture"]),
    (3, Intent.DESCRIBE_SCREEN,  ["what is on my screen", "analyze my screen", "describe the screen", "what am i looking at"]),
    # Media
    (3, Intent.PLAY_MUSIC,       ["play music", "play song", "play some music", "start music", "resume music", "resume song"]),
    (3, Intent.PAUSE_MUSIC,      ["pause music", "stop music", "pause song"]),
    (3, Intent.SKIP_TRACK,       ["skip", "next song", "next track", "skip track", "previous song"]),
    # Volume
    (3, Intent.VOLUME_UP,        ["volume up", "turn up volume", "louder", "increase volume"]),
    (3, Intent.VOLUME_DOWN,      ["volume down", "turn down volume", "quieter", "decrease volume", "lower volume"]),
    (3, Intent.VOLUME_MUTE,      ["mute", "unmute", "toggle mute"]),
    # Timers
    (3, Intent.SET_TIMER,        ["set a timer", "set timer", "start a timer", "timer for"]),
    (3, Intent.SET_ALARM,        ["set an alarm", "set alarm", "wake me at", "alarm at", "alarm for"]),
    (3, Intent.SET_REMINDER,     ["remind me", "set a reminder", "reminder to", "remind"]),
    # Calendar
    (3, Intent.CALENDAR_READ,    ["my schedule", "what's on my calendar", "calendar today", "my agenda", "upcoming events"]),
    (3, Intent.CALENDAR_ADD,     ["add to calendar", "schedule meeting", "book appointment", "add event"]),
    # Weather
    (3, Intent.GET_WEATHER,      ["weather", "temperature outside", "will it rain", "forecast"]),
    # News
    (3, Intent.GET_NEWS,         ["news", "headlines", "top stories", "what's happening", "latest news"]),
    # Math
    (6, Intent.CALCULATE,        ["calculate", "what is", "compute", "how much is", "evaluate"]),
    # Translate
    (3, Intent.TRANSLATE,        ["translate", "how do you say", "what does", "in spanish", "in french", "in german", "in japanese"]),
    # Email
    (4, Intent.SEND_EMAIL,       ["send email", "send an email", "email to", "write an email", "compose email"]),
    (4, Intent.READ_EMAIL,       ["read my email", "check email", "my inbox", "new emails", "read emails"]),
    # Files
    (4, Intent.FIND_FILE,        ["find file", "search for file", "where is the file", "locate file", "find the"]),
    (4, Intent.READ_FILE,        ["read file", "open file", "show me the file", "what's in", "read the"]),
    (4, Intent.WRITE_FILE,       ["write to file", "create file", "save file", "write file"]),
    (4, Intent.DELETE_FILE,      ["delete file", "remove file", "trash the file"]),
    # Notes
    (4, Intent.TAKE_NOTE,        ["take a note", "note that", "jot down", "write down", "add a note"]),
    (4, Intent.READ_NOTE,        ["read my notes", "show my notes", "what notes", "open notes"]),
    # Knowledge base
    (4, Intent.KB_QUERY,         ["what do you know about", "remember that", "do you know about", "recall"]),
    (4, Intent.KB_ADD,           ["remember this", "save this to memory", "learn this", "add to knowledge"]),
    # Clipboard
    (4, Intent.CLIPBOARD_READ,   ["read clipboard", "what's in clipboard", "paste", "what did i copy"]),
    (4, Intent.CLIPBOARD_WRITE,  ["copy to clipboard", "save to clipboard"]),
    # Code
    (5, Intent.CODE_EXECUTE,     ["run code", "execute code", "run this script", "run python", "execute script"]),
    # Computer use
    (5, Intent.COMPUTER_USE,     ["do it on screen", "control my desktop", "use the computer to", "click on", "type into the"]),
    # Agents
    (5, Intent.SWARM_TASK,       ["swarm", "multiple agents", "research and write", "team of agents"]),
    (5, Intent.AUTONOMOUS_AGENT, ["automate", "organize my", "arrange my", "on my desktop", "in that folder", "install software", "fill out this form"]),
    # System status
    (5, Intent.SYSTEM_STATUS,    ["system status", "cpu usage", "ram usage", "memory usage", "disk space", "how hot", "system info", "battery"]),
    # Search (low priority — very broad)
    (9, Intent.SEARCH_WEB,       ["search", "look up", "google", "find out", "tell me about"]),
]

# Sort by priority once at import time
_KEYWORD_RULES.sort(key=lambda r: r[0])


# ── Fuzzy matching helpers ────────────────────────────────────────────────────

try:
    from rapidfuzz import fuzz as _fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False


# ── Slot patterns ─────────────────────────────────────────────────────────────

_SLOT_PATTERNS: Dict[str, re.Pattern] = {
    # application name — word(s) after open/launch/close/exit/quit/run
    "app": re.compile(
        r'\b(?:open|launch|start|run|close|exit|quit|kill|terminate)\s+([a-zA-Z0-9_\-\.\s]{1,40}?)(?:\s+and\b|$)',
        re.IGNORECASE,
    ),
    # URL / domain
    "url": re.compile(
        r'(?:go to|navigate to|browse to|visit|open)\s+((?:https?://)?[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)',
        re.IGNORECASE,
    ),
    # search query — text after "search for / look up / google"
    "query": re.compile(
        r'\b(?:search(?:\s+for)?|look\s+up|google|find\s+out\s+about|tell\s+me\s+about)\s+(.+)',
        re.IGNORECASE,
    ),
    # duration in minutes/seconds/hours
    "duration": re.compile(
        r'(\d+)\s*(second|seconds|sec|minute|minutes|min|hour|hours|hr)',
        re.IGNORECASE,
    ),
    # specific time (HH:MM or "5 pm")
    "time": re.compile(
        r'\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b',
        re.IGNORECASE,
    ),
    # email recipient
    "recipient": re.compile(
        r'\b(?:to|email)\s+([A-Za-z][A-Za-z\s]{1,30}?)(?:\s+about|\s+saying|\s+with|\s*$)',
        re.IGNORECASE,
    ),
    # file name / path
    "file": re.compile(
        r'\b(?:file|document|called|named|the)\s+(["\']?[A-Za-z0-9_\-\.\s]{1,60}?["\']?)(?:\s|$)',
        re.IGNORECASE,
    ),
    # volume level (percentage)
    "volume_level": re.compile(
        r'(\d{1,3})\s*(?:percent|%)',
        re.IGNORECASE,
    ),
    # note content
    "note_content": re.compile(
        r'\b(?:note that|jot down|write down|remember|take a note[:\s])\s*[:\-]?\s*(.+)',
        re.IGNORECASE,
    ),
}


# ── Main detector ─────────────────────────────────────────────────────────────

class IntentDetector:
    """
    Three-tier intent classification:
    1. Keyword phrase matching (fast, deterministic)
    2. Fuzzy string matching (handles typos/paraphrases)
    3. Sentence embedding similarity (requires sentence-transformers)

    Example usage
    ─────────────
        detector = IntentDetector()
        intent = detector.detect_intent("open chrome and navigate to github.com")
        slots  = detector.extract_slots("open chrome and navigate to github.com", intent)
        # intent → Intent.OPEN_APP
        # slots  → {"app": "chrome", "url": "github.com"}
    """

    # Fuzzy threshold (0–100)
    _FUZZY_THRESHOLD = 70

    def __init__(self, semantic_enabled: bool = True):
        self._semantic_enabled = semantic_enabled
        self._semantic_clf = None   # lazy-loaded
        if semantic_enabled:
            try:
                from sentinel.core.semantic_intent import get_semantic_classifier
                self._semantic_clf = get_semantic_classifier()
            except (ImportError, Exception) as exc:
                logger.warning("Semantic classifier unavailable (falling back to keyword/fuzzy): %s", exc)
                self._semantic_enabled = False

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_intent(self, command: str) -> Intent:
        """Return the best-matching Intent for the given pre-processed command."""
        intent, _ = self._classify(command)
        return intent

    def detect_with_confidence(self, command: str) -> Tuple[Intent, float]:
        """Return (Intent, confidence_0_to_1) for the given command."""
        return self._classify(command)

    def extract_entity(self, command: str, intent: Intent) -> Optional[str]:
        """
        Legacy single-entity extraction (kept for backwards compatibility).
        Returns the most relevant slot value for the given intent.
        """
        slots = self.extract_slots(command, intent)
        # Priority order of slots per intent
        _priority = {
            Intent.OPEN_APP:    ["app"],
            Intent.CLOSE_APP:   ["app"],
            Intent.BROWSE_URL:  ["url", "app"],
            Intent.SEARCH_WEB:  ["query"],
            Intent.GET_WEATHER: ["query"],
            Intent.SET_TIMER:   ["duration"],
            Intent.SET_ALARM:   ["time"],
            Intent.SEND_EMAIL:  ["recipient"],
            Intent.FIND_FILE:   ["file"],
            Intent.READ_FILE:   ["file"],
            Intent.DELETE_FILE: ["file"],
            Intent.TAKE_NOTE:   ["note_content", "query"],
            Intent.TRANSLATE:   ["query"],
            Intent.CALCULATE:   ["query"],
        }
        for slot_key in _priority.get(intent, ["app", "query"]):
            value = slots.get(slot_key)
            if value:
                return value.strip()
        # Fallback: remove the matched trigger phrase and return remnant
        return self._extract_remnant(command, intent)

    def extract_slots(self, command: str, intent: Intent) -> Dict[str, str]:
        """
        Extract all named slots from the command.
        Returns dict like {"app": "chrome", "url": "github.com"}.
        """
        slots: Dict[str, str] = {}
        for slot_name, pattern in _SLOT_PATTERNS.items():
            match = pattern.search(command)
            if match:
                slots[slot_name] = match.group(1).strip()
        return slots

    # ── Internal classification ───────────────────────────────────────────────

    def _classify(self, command: str) -> Tuple[Intent, float]:
        cmd = (command or "").lower().strip()
        if not cmd:
            return Intent.UNKNOWN, 0.0

        # Tier 1: keyword match
        intent = self._keyword_match(cmd)
        if intent is not None:
            return intent, 1.0

        # Tier 2: fuzzy match
        intent, score = self._fuzzy_match(cmd)
        if intent is not None:
            return intent, score / 100.0

        # Tier 3: semantic embedding
        if self._semantic_clf:
            try:
                label, sim = self._semantic_clf.predict(cmd)
                if label is not None:
                    try:
                        return Intent(label), sim
                    except ValueError:
                        logger.debug("Semantic label '%s' not in Intent enum.", label)
            except Exception as exc:
                logger.debug("Semantic classification error: %s", exc)

        return Intent.UNKNOWN, 0.0

    def _keyword_match(self, cmd: str) -> Optional[Intent]:
        """Check each rule's phrase list; return the first match in priority order."""
        for _priority, intent, phrases in _KEYWORD_RULES:
            for phrase in phrases:
                if phrase in cmd:
                    return intent
        return None

    def _fuzzy_match(self, cmd: str) -> Tuple[Optional[Intent], float]:
        """Score each intent's representative phrase against the command."""
        if not _RAPIDFUZZ_AVAILABLE:
            return None, 0.0

        best_intent: Optional[Intent] = None
        best_score: float = 0.0

        # Build a representative phrase per intent (first phrase of its rule)
        representatives: Dict[Intent, str] = {}
        for _p, intent, phrases in _KEYWORD_RULES:
            if intent not in representatives and phrases:
                representatives[intent] = phrases[0]

        for intent, phrase in representatives.items():
            score = _fuzz.partial_ratio(phrase, cmd)
            if score > best_score:
                best_score = score
                best_intent = intent

        if best_score >= self._FUZZY_THRESHOLD:
            return best_intent, best_score
        return None, best_score

    def _extract_remnant(self, command: str, intent: Intent) -> Optional[str]:
        """
        Fallback entity extraction: remove the trigger phrase and return
        the remainder (cleaned up).
        """
        cmd = command.lower()
        for _p, kw_intent, phrases in _KEYWORD_RULES:
            if kw_intent != intent:
                continue
            for phrase in phrases:
                if phrase.strip() in cmd:
                    remnant = cmd.replace(phrase.strip(), "").strip()
                    # Remove common filler
                    for filler in ("please", "now", "the", "a", "an"):
                        remnant = re.sub(r'\b' + filler + r'\b', '', remnant).strip()
                    if remnant:
                        return remnant
        return None
