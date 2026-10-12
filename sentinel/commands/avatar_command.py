"""
sentinel/commands/avatar_command.py
────────────────────────────────────
Voice Command Plugin for Holographic Avatar Control.

Handles: "show avatar", "hide avatar", "avatar cute anger", "avatar flirt",
         "avatar focus", "move avatar", "resize avatar".
"""

import logging
import re

from sentinel.ar.holographic_avatar import (
    AvatarEmotion,
    HolographicAvatarEngine,
)

logger = logging.getLogger("AvatarCommand")


# Map voice-friendly words to AvatarEmotion
_EMOTION_ALIASES = {
    "idle": AvatarEmotion.IDLE,
    "normal": AvatarEmotion.IDLE,
    "speaking": AvatarEmotion.SPEAKING,
    "talking": AvatarEmotion.SPEAKING,
    "listening": AvatarEmotion.LISTENING,
    "cute anger": AvatarEmotion.CUTE_ANGER,
    "angry": AvatarEmotion.CUTE_ANGER,
    "pout": AvatarEmotion.CUTE_ANGER,
    "tsundere": AvatarEmotion.CUTE_ANGER,
    "flirty": AvatarEmotion.FLIRTY,
    "flirt": AvatarEmotion.FLIRTY,
    "romantic": AvatarEmotion.FLIRTY,
    "focused": AvatarEmotion.FOCUSED,
    "focus": AvatarEmotion.FOCUSED,
    "working": AvatarEmotion.FOCUSED,
    "empathic": AvatarEmotion.EMPATHIC,
    "caring": AvatarEmotion.EMPATHIC,
    "comfort": AvatarEmotion.EMPATHIC,
    "excited": AvatarEmotion.EXCITED,
    "happy": AvatarEmotion.EXCITED,
}


class AvatarCommand:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.avatar = HolographicAvatarEngine()

    def execute(self, command: str, entity: str | None = None) -> str:
        cmd = command.lower().strip()

        # 1. Show / activate avatar
        if any(k in cmd for k in ("show avatar", "activate avatar", "display avatar", "summon avatar", "avatar on")):
            self.avatar.show()
            self.avatar.start_render_loop()
            return "✨ Holographic avatar activated! I'm now visible on your desktop with real-time lip-sync and particle effects."

        # 2. Hide / dismiss avatar
        if any(k in cmd for k in ("hide avatar", "dismiss avatar", "avatar off", "close avatar", "deactivate avatar")):
            self.avatar.stop_render_loop()
            self.avatar.hide()
            return "Avatar hidden. I'll still be listening in the background."

        # 3. Emotion change
        for alias, emotion in _EMOTION_ALIASES.items():
            if alias in cmd:
                result = self.avatar.set_emotion(emotion)
                expr = result["expression"]
                return (
                    f"Avatar emotion set to {emotion.value}. Expression: {expr}. Core glow: RGB{result['core_color']}."
                )

        # 4. Move / reposition
        pos_match = re.search(r"move\s+avatar\s+(?:to\s+)?(\d+)\s*[,x]\s*(\d+)", cmd)
        if pos_match:
            x, y = int(pos_match.group(1)), int(pos_match.group(2))
            self.avatar.set_position(x, y)
            return f"Avatar repositioned to ({x}, {y})."

        # 5. Resize
        size_match = re.search(r"(?:resize|scale)\s+avatar\s+(?:to\s+)?(\d+)", cmd)
        if size_match:
            size = int(size_match.group(1))
            actual = self.avatar.set_size(size)
            return f"Avatar resized to {actual}px diameter."

        # 6. Opacity
        opacity_match = re.search(r"avatar\s+opacity\s+(\d+)", cmd)
        if opacity_match:
            pct = int(opacity_match.group(1))
            actual = self.avatar.set_opacity(pct / 100.0)
            return f"Avatar opacity set to {int(actual * 100)}%."

        # 7. Status
        if "avatar status" in cmd or "avatar info" in cmd:
            status = self.avatar.get_status()
            return (
                f"Avatar: {'Visible' if status['visible'] else 'Hidden'} | "
                f"Emotion: {status['emotion']} | Position: {status['position']} | "
                f"Size: {status['size']}px | Particles: {status['particle_count']}"
            )

        return "Avatar command not recognized. Try: show avatar, hide avatar, avatar cute anger, avatar status."
