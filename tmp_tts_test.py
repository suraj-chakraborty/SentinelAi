import logging
from sentinel.app.voice import speak

logging.basicConfig(level=logging.INFO)
print("Testing TTS...")
speak("Sentinel voice system check. Can you hear me?", block=True)
print("Test complete.")
