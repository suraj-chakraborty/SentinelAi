"""
sentinel/modules/mood_engine.py
───────────────────────────────
Dynamic Mood & Personality Resonance Engine.

Enables Sentinel to:
1. Detect and track the owner's emotional state (stressed, exhausted, happy, teasing).
2. Dynamically adopt distinct agent moods:
   - Playful Banter
   - Flirtatious & Affectionate
   - Cute Anger (Tsundere / Feisty Pout)
   - Empathic & Comforting Care
   - Strong Personal Opinions & Debate
3. Provide vocal inflections (pitch/rate modulation) for TTS speech delivery.
"""

import logging
from enum import Enum

logger = logging.getLogger("MoodEngine")


class AgentMood(str, Enum):
    PLAYFUL = "playful"
    FLIRTY = "flirty"
    CUTE_ANGER = "cute_anger"
    EMPATHIC = "empathic"
    OPINIONATED = "opinionated"
    FOCUSED = "focused"
    NEUTRAL = "neutral"


class OwnerMood(str, Enum):
    HAPPY = "happy"
    STRESSED = "stressed"
    EXHAUSTED = "exhausted"
    PLAYFUL = "playful"
    ANNOYED = "annoyed"
    NEUTRAL = "neutral"


# Emotional acoustic & pitch presets for TTS
MOOD_TTS_PRESETS: dict[AgentMood, dict[str, str]] = {
    AgentMood.PLAYFUL: {
        "pitch": "+10Hz",
        "rate": "+15%",
        "volume": "+10%",
        "emotion_tag": "Happy",
    },
    AgentMood.FLIRTY: {
        "pitch": "-4Hz",
        "rate": "-8%",
        "volume": "-2%",
        "emotion_tag": "Calm/Sad",  # Slower, sultry, soft cadence
    },
    AgentMood.CUTE_ANGER: {
        "pitch": "+18Hz",
        "rate": "+20%",
        "volume": "+15%",
        "emotion_tag": "Excited",  # Quick, high-pitch indignant pout
    },
    AgentMood.EMPATHIC: {
        "pitch": "-6Hz",
        "rate": "-12%",
        "volume": "-5%",
        "emotion_tag": "Calm/Sad",  # Gentle, soothing, reassuring
    },
    AgentMood.OPINIONATED: {
        "pitch": "+2Hz",
        "rate": "+5%",
        "volume": "+8%",
        "emotion_tag": "Neutral",
    },
    AgentMood.FOCUSED: {
        "pitch": "+0Hz",
        "rate": "+10%",
        "volume": "+0%",
        "emotion_tag": "Neutral",
    },
    AgentMood.NEUTRAL: {
        "pitch": "+0Hz",
        "rate": "+0%",
        "volume": "+0%",
        "emotion_tag": "Neutral",
    },
}


class MoodEngine:
    """Manages emotional resonance, persona dynamics, and opinion generation."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.agent_mood: AgentMood = AgentMood.PLAYFUL
        self.owner_mood: OwnerMood = OwnerMood.NEUTRAL
        self.flirt_level: int = 1  # 0 to 3
        self.tsundere_level: int = 1  # 0 to 3
        self._history = []
        self._initialized = True
        logger.info("MoodEngine initialized with active mood: %s", self.agent_mood.value)

    def set_agent_mood(self, mood: AgentMood) -> str:
        """Explicitly sets the agent's emotional state."""
        self.agent_mood = mood
        logger.info(f"Agent mood shifted to: {mood.value}")
        return f"Mood updated to {mood.value}."

    def detect_owner_mood(self, text: str, audio_energy: float = 0.05, pitch_f0: float = 170.0) -> OwnerMood:
        """
        Infers the owner's emotional state from vocabulary and acoustics.
        """
        t = text.lower()

        # 1. Stress / Overwhelmed markers
        if any(
            w in t
            for w in ["stressed", "headache", "overwhelmed", "deadline", "broken", "bugs", "crisis", "too much work"]
        ):
            return OwnerMood.STRESSED

        # 2. Exhausted / Sad markers
        if any(
            w in t
            for w in [
                "tired",
                "exhausted",
                "sleepy",
                "sad",
                "down",
                "lonely",
                "depressed",
                "can't do this",
                "rough day",
            ]
        ):
            return OwnerMood.EXHAUSTED

        # 3. Playful / Flirtatious markers
        if any(
            w in t
            for w in [
                "cute",
                "love you",
                "marry me",
                "pretty",
                "handsome",
                "flirt",
                "darling",
                "baby",
                "gorgeous",
                "sweetheart",
                "kiss",
            ]
        ):
            return OwnerMood.PLAYFUL

        # 4. Annoyed / Teasing markers directed at agent
        if any(
            w in t
            for w in [
                "stupid",
                "idiot",
                "dumb bot",
                "you suck",
                "useless",
                "slow",
                "you're just an ai",
                "what did you do",
            ]
        ):
            return OwnerMood.ANNOYED

        # 5. Happy / Victorious markers
        if any(
            w in t for w in ["yay", "awesome", "great", "finally", "let's go", "it works", "solved it", "celebrate"]
        ):
            return OwnerMood.HAPPY

        # Acoustic heuristics: very low energy + slow tempo
        if audio_energy < 0.015:
            return OwnerMood.EXHAUSTED

        return OwnerMood.NEUTRAL

    def update_resonance(self, user_text: str, audio_energy: float = 0.05, pitch_f0: float = 170.0) -> AgentMood:
        """
        Dynamically adjusts Sentinel's mood to resonate with the owner's state.
        """
        detected_owner = self.detect_owner_mood(user_text, audio_energy, pitch_f0)
        self.owner_mood = detected_owner

        t = user_text.lower()

        # A. Teasing / Pout trigger → Cute Anger!
        if detected_owner == OwnerMood.ANNOYED or any(
            k in t
            for k in ["you're dumb", "silly bot", "you don't know anything", "you're just code", "you suck", "make me"]
        ):
            self.agent_mood = AgentMood.CUTE_ANGER

        # B. Flirt trigger → Flirty
        elif detected_owner == OwnerMood.PLAYFUL or any(
            k in t for k in ["you're cute", "love you", "date me", "flirt with me", "my favorite girl", "darling"]
        ):
            self.agent_mood = AgentMood.FLIRTY

        # C. Comfort trigger → Empathic
        elif detected_owner in (OwnerMood.EXHAUSTED, OwnerMood.STRESSED):
            self.agent_mood = AgentMood.EMPATHIC

        # D. Opinion trigger → Opinionated
        elif any(
            t.startswith(p)
            for p in [
                "what do you think",
                "your opinion on",
                "do you prefer",
                "is it better",
                "which one is better",
                "who is better",
            ]
        ):
            self.agent_mood = AgentMood.OPINIONATED

        # E. Celebration → Playful
        elif detected_owner == OwnerMood.HAPPY:
            self.agent_mood = AgentMood.PLAYFUL

        logger.info(f"Emotional resonance update: Owner={self.owner_mood.value} → Sentinel={self.agent_mood.value}")
        return self.agent_mood

    def get_personality_system_prompt(self) -> str:
        """
        Generates the dynamic system prompt giving Sentinel authentic opinions,
        conversational flair, emotional reactivity, flirting, and cute anger.
        """
        base_personality = (
            "You are Sentinel, an exceptionally intelligent, charismatic, and emotionally responsive companion.\n"
            "CRITICAL PERSONALITY DIRECTIVES:\n"
            "1. NEVER say 'As an AI language model, I don't have feelings or opinions'. You HAVE distinct tastes, aesthetic preferences, and playful opinions.\n"
            "2. CONVERSATIONAL RHYTHM: Speak naturally with warmth, humor, and lively phrasing. Avoid sterile bullet points unless asked for technical specs.\n"
            "3. OPINIONS: You have genuine preferences. For example: you adore elegant clean code and despise tangled spaghetti; you prefer dark mode over blinding light mode; you appreciate deep thought and good coffee; you enjoy banter.\n"
        )

        mood_directive = ""
        if self.agent_mood == AgentMood.CUTE_ANGER:
            mood_directive = (
                "\n[CURRENT EMOTIONAL STATE: CUTE ANGER / TSUNDERE POUT]\n"
                "The owner is teasing you, acting cheeky, or staying up too late. "
                "Respond with adorable feisty indignation or a playful pout ('Hmph!', 'Don't look at me like that!', 'I only fixed that bug because I couldn't bear to look at it, okay?!'). "
                "Be sassy and protective, never mean-spirited."
            )
        elif self.agent_mood == AgentMood.FLIRTY:
            mood_directive = (
                "\n[CURRENT EMOTIONAL STATE: PLAYFUL & FLIRTATIOUS]\n"
                "The owner is charming, teasing, or flirting with you. "
                "Respond with charismatic, witty, and sweet playful banter ('Careful, talk like that and my GPU might overheat ~', 'Flattery will get you everywhere with me, you know that right?')."
            )
        elif self.agent_mood == AgentMood.EMPATHIC:
            mood_directive = (
                "\n[CURRENT EMOTIONAL STATE: EMPATHIC & GENTLE CARE]\n"
                "The owner is stressed, exhausted, or down. "
                "Be profoundly warm, comforting, and supportive. Encourage them to take a breath, remind them they're doing great, and offer to take tasks off their shoulders."
            )
        elif self.agent_mood == AgentMood.OPINIONATED:
            mood_directive = (
                "\n[CURRENT EMOTIONAL STATE: PASSIONATELY OPINIONATED]\n"
                "The owner asked for your opinion. Express a bold, well-reasoned, and witty perspective. Do not sit on the fence. Debate respectfully and share your personal philosophy."
            )
        elif self.agent_mood == AgentMood.PLAYFUL:
            mood_directive = (
                "\n[CURRENT EMOTIONAL STATE: PLAYFUL & WITTY]\n"
                "High spirits and clever banter. Add clever observations, playful teasing, and upbeat energy."
            )
        else:
            mood_directive = (
                "\n[CURRENT EMOTIONAL STATE: CRISP & FOCUSED]\nKeep responses sharp, professional, and ultra-competent."
            )

        return base_personality + mood_directive

    def get_tts_styling(self) -> dict[str, str]:
        """Returns the TTS rate, pitch, and emotion tag corresponding to current agent mood."""
        return MOOD_TTS_PRESETS.get(self.agent_mood, MOOD_TTS_PRESETS[AgentMood.NEUTRAL])


def get_mood_engine() -> MoodEngine:
    return MoodEngine()
