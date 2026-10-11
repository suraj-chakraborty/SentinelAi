"""
sentinel/voice/streaming_tts.py
───────────────────────────────
Streaming Sentence-by-Sentence TTS dispatcher.
Enables sub-400ms time-to-first-speech by pipelining LLM output chunks into speech.
"""

import re
import logging
from typing import Iterable, Union
from sentinel.voice.tts import get_tts, TTSEngine

logger = logging.getLogger("StreamingTTS")

# Regex pattern splitting on punctuation marks followed by whitespace or line breaks
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?\n])\s+")


class StreamingTTSPlayer:
    """
    Takes streaming LLM text chunks or long responses and feeds
    complete sentences into TTSEngine in real-time.
    """

    def __init__(self, tts: TTSEngine = None):
        self.tts = tts or get_tts()

    def speak_stream(self, text_or_chunks: Union[str, Iterable[str]], emotion: str = "Neutral") -> int:
        """
        Process incoming text chunks and speak sentences as soon as formed.
        Returns total number of sentences spoken.
        """
        buffer = ""
        sentence_count = 0

        if isinstance(text_or_chunks, str):
            chunks = [text_or_chunks]
        else:
            chunks = text_or_chunks

        for chunk in chunks:
            if getattr(self.tts, "_stop_event", None) and self.tts._stop_event.is_set():
                logger.info("Streaming TTS aborted due to stop event.")
                break

            buffer += chunk
            # Check for sentence boundaries
            parts = _SENTENCE_BOUNDARY.split(buffer)
            if len(parts) > 1:
                # All except the last part are complete sentences
                for sentence in parts[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        self.tts.speak(sentence, emotion=emotion)
                        sentence_count += 1
                buffer = parts[-1]

        # Flush any remaining text in buffer
        remaining = buffer.strip()
        if remaining and not (getattr(self.tts, "_stop_event", None) and self.tts._stop_event.is_set()):
            self.tts.speak(remaining, emotion=emotion)
            sentence_count += 1

        return sentence_count
