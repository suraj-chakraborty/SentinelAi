"""
sentinel/app/voice_enroll.py
─────────────────────────────
Voice Enrollment Wizard.
Guides the user through recording their reference voice profile.
"""

import os
import time
import logging
import numpy as np
from sentinel.app.audio import record_until_silence, compute_embedding
from sentinel.app.config import REFERENCE_WAV, OWNER_EMBED_PATH

logger = logging.getLogger("VoiceEnrollment")

def start_enrollment(on_status=None):
    """
    Step-by-step voice registration process.
    """
    from sentinel.app.voice import speak

    def log(msg):
        logger.info(msg)
        if on_status:
            on_status(msg)

    log("Starting Voice Enrollment...")
    
    # 1. Introduction
    speak("Welcome to Sentinel Voice Registration. I need to calibrate my ears to your unique voice.")
    time.sleep(0.5)
    
    # 2. Instructions
    phrase = "The quick brown fox jumps over the lazy dog"
    speak(f"Please say the following phrase clearly: {phrase}")
    log(f"Please say: '{phrase}'")
    
    # 3. Recording
    time.sleep(1.0) # wait for user to prepare
    try:
        path, had_speech = record_until_silence(REFERENCE_WAV, max_duration=10)
        
        if not had_speech:
            speak("I didn't hear anything. Let's try again when you're ready.")
            log("Enrollment failed: No speech detected.")
            return False
            
        log("Analyzing voice characteristics...")
        
        # 4. Processing
        embed = compute_embedding(path)
        if embed is not None and np.any(embed):
            np.save(OWNER_EMBED_PATH, embed)
            log("Voice profile saved successfully.")
            speak("Calibration complete. Your voice profile is now securely stored. I will only respond to you.")
            return True
        else:
            log("Enrollment failed: Analysis error.")
            speak("I encountered an error while analyzing your voice. Please try again.")
            return False
            
    except Exception as e:
        logger.error(f"Enrollment Error: {e}")
        log(f"Enrollment Error: {str(e)[:30]}")
        speak("I'm sorry, something went wrong during the enrollment process.")
        return False
