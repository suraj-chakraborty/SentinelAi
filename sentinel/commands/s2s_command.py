"""
sentinel/commands/s2s_command.py
────────────────────────────────
Voice Command Plugin for Native Speech-to-Speech Streaming.
Handles "start live speech", "translate live to <lang>", and "acoustic mirroring".
"""

import logging

from sentinel.voice.live_speech_stream import LiveSpeechStream

logger = logging.getLogger("S2SCommand")


class S2SCommand:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.stream = LiveSpeechStream()

    def execute(self, command: str, entity: str | None = None) -> str:
        cmd_lower = command.lower()

        # 1. Connect / Start live streaming
        if any(k in cmd_lower for k in ("start live speech", "enable live stream", "fast voice mode", "sub 200ms")):
            self.stream.connect()
            return "Live Speech-to-Speech stream activated. Sub-200ms latency and acoustic mirroring enabled."

        # 2. In-Call Live Translation
        if "translate live" in cmd_lower or "in-call translation" in cmd_lower:
            target_code = "es"
            target_name = "Spanish"
            if "japanese" in cmd_lower:
                target_code, target_name = "ja", "Japanese"
            elif "german" in cmd_lower:
                target_code, target_name = "de", "German"
            elif "french" in cmd_lower:
                target_code, target_name = "fr", "French"
            elif "spanish" in cmd_lower:
                target_code, target_name = "es", "Spanish"

            self.stream.set_translation_target(target_code)
            self.stream.connect()
            return f"Live in-call translation enabled. Outgoing speech translating to {target_name.upper()} ({target_code.upper()}) in real time."

        # 3. Disconnect
        if any(k in cmd_lower for k in ("stop live speech", "disable live stream")):
            self.stream.disconnect()
            return "Live speech streaming deactivated. Returned to standard mode."

        return "Speech-to-speech command not recognized."
