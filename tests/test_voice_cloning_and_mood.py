"""
tests/test_voice_cloning_and_mood.py
────────────────────────────────────
Test suite for custom voice recording/cloning, emotional resonance,
opinions, flirting, and cute anger dynamics.
"""

import os
import sys
import tempfile
import wave
import numpy as np
from unittest.mock import MagicMock, patch

# Ensure sentinel package is resolvable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentinel.voice.voice_cloner import VoiceProfileManager
from sentinel.modules.mood_engine import MoodEngine, AgentMood, OwnerMood, get_mood_engine
from sentinel.commands.voice_profile_command import VoiceProfileCommand
from sentinel.commands.mood_command import MoodCommand
from sentinel.core.intent_detector import IntentDetector, Intent
from sentinel.core.command_router import CommandRouter


class TestVoiceCloningAndMoodDynamics:

    def test_voice_cloner_acoustic_profiling(self):
        """Test acoustic feature extraction and voice profile generation."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav_path = tf.name

        try:
            # Generate 1-second synthetic 16kHz sine wave audio
            sample_rate = 16000
            t = np.linspace(0, 1.0, sample_rate, False)
            tone = (np.sin(2 * np.pi * 220 * t) * 10000).astype(np.int16)

            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(tone.tobytes())

            manager = VoiceProfileManager()
            prof = manager.analyze_and_create_profile(wav_path, profile_name="Synthetic Test Voice")

            assert prof["is_custom"] is True
            assert "base_voice" in prof
            assert "pitch_offset" in prof
            assert prof["detected_f0"] > 0
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def test_voice_cloner_command_and_reset(self):
        """Test VoiceProfileCommand execution and factory reset."""
        cmd = VoiceProfileCommand()
        reset_res = cmd.execute("reset agent voice")
        assert "default" in reset_res.lower()

        status_res = cmd.execute("what is your voice profile")
        assert "voice profile" in status_res.lower()

    def test_mood_engine_owner_detection(self):
        """Test detection of owner emotional states."""
        engine = MoodEngine()

        # Stressed
        assert engine.detect_owner_mood("I am so stressed with these production bugs and deadlines") == OwnerMood.STRESSED

        # Exhausted
        assert engine.detect_owner_mood("I am exhausted and had a really rough day") == OwnerMood.EXHAUSTED

        # Playful / Flirtatious
        assert engine.detect_owner_mood("You are so cute darling, marry me") == OwnerMood.PLAYFUL

        # Annoyed / Teasing agent
        assert engine.detect_owner_mood("You stupid bot, what did you just do") == OwnerMood.ANNOYED

        # Happy
        assert engine.detect_owner_mood("Yay it finally works! Let's celebrate!") == OwnerMood.HAPPY

    def test_mood_engine_resonance_and_cute_anger(self):
        """Test emotional resonance: cute anger, flirting, and opinions."""
        engine = get_mood_engine()

        # Teasing → Cute Anger
        mood_anger = engine.update_resonance("You're a silly bot and you don't know anything!")
        assert mood_anger == AgentMood.CUTE_ANGER

        prompt_anger = engine.get_personality_system_prompt()
        assert "CUTE ANGER" in prompt_anger
        assert "Hmph!" in prompt_anger

        # Flirting → Flirty
        mood_flirt = engine.update_resonance("I love you so much darling, you're my favorite")
        assert mood_flirt == AgentMood.FLIRTY

        prompt_flirt = engine.get_personality_system_prompt()
        assert "FLIRTATIOUS" in prompt_flirt

        # Opinion Query → Opinionated
        mood_op = engine.update_resonance("What do you think of Python vs Rust?")
        assert mood_op == AgentMood.OPINIONATED

        # TTS Styling
        styling = engine.get_tts_styling()
        assert "pitch" in styling
        assert "rate" in styling

    def test_mood_command_execution(self):
        """Test explicit mood shifts via MoodCommand."""
        cmd = MoodCommand()

        # Cute anger
        anger_resp = cmd.execute("be angry with me")
        assert "Hmph!" in anger_resp

        # Flirting
        flirt_resp = cmd.execute("flirt with me")
        assert "GPU" in flirt_resp or "blush" in flirt_resp

        # Empathic
        care_resp = cmd.execute("comfort me please")
        assert "right here with you" in care_resp

    def test_intent_routing_for_voice_and_mood(self):
        """Test intent classification and routing to voice and mood plugins."""
        detector = IntentDetector()
        assert detector.detect_intent("record voice sample for agent") == Intent.VOICE_CLONE
        assert detector.detect_intent("reset agent voice to default") == Intent.VOICE_CLONE
        assert detector.detect_intent("flirt with me right now") == Intent.AGENT_MOOD
        assert detector.detect_intent("show cute anger") == Intent.AGENT_MOOD

        mock_orch = MagicMock()
        mock_orch._safe_llm_call.return_value = "I definitely prefer clean modular code."
        router = CommandRouter(mock_orch)

        # Dispatch voice clone command
        voice_out = router.run_command("reset agent voice")
        assert "default" in voice_out.lower()

        # Dispatch mood command
        mood_out = router.run_command("show cute anger")
        assert "Hmph!" in mood_out
