"""
sentinel/app/audio.py
──────────────────────
Audio capture, VAD, and liveness check.
"""

import os
import time
import wave
import numpy as np
import pyaudio
import soundfile as sf
import logging
from typing import Optional, Tuple

from sentinel.app.config import (
    REFERENCE_WAV, LIVENESS_WAV, OWNER_EMBED_PATH,
    ensure_appdata_dir
)

logger = logging.getLogger("SentinelAudio")

# ── Low-Level Capture ────────────────────────────────────────────────────────

def record_until_silence(filename: str, 
                         max_duration=25, 
                         samplerate=16000, 
                         channels=1, 
                         frames_per_buffer=1024, 
                         min_duration=1.5, 
                         start_timeout=5.0, 
                         silence_threshold=0.006, 
                         speech_threshold=0.005, 
                         silence_duration=0.6, 
                         on_amp=None) -> Tuple[str, bool]:
    """
    Record audio until the user stops speaking.
    Returns (filename, had_speech).
    """
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(format=pyaudio.paInt16,
                         channels=channels,
                         rate=samplerate,
                         input=True,
                         frames_per_buffer=frames_per_buffer)
    except Exception as e:
        logger.error(f"Mic access failed: {e}")
        pa.terminate()
        return filename, False

    frames = []
    start_t = time.time()
    quiet_t = 0.0
    had_speech = False
    peak_amp = 0.0
    
    # Simple median baseline
    baseline_frames = int(max(1, (samplerate / frames_per_buffer) * 0.5))
    baseline_vals = []

    try:
        while True:
            data = stream.read(frames_per_buffer, exception_on_overflow=False)
            frames.append(data)
            samples = np.frombuffer(data, dtype=np.int16)
            amp = float(np.mean(np.abs(samples))) / 32768.0
            
            if amp > peak_amp: peak_amp = amp
            
            now = time.time()
            elapsed = now - start_t
            
            if len(baseline_vals) < baseline_frames:
                baseline_vals.append(amp)
                continue
            
            base = np.median(baseline_vals)
            sp_thr = max(speech_threshold, base * 2.0)
            si_thr = max(silence_threshold, base * 1.2)
            
            if amp < si_thr:
                quiet_t += frames_per_buffer / float(samplerate)
            else:
                quiet_t = 0.0
            
            if amp >= sp_thr:
                had_speech = True
                
            if on_amp:
                try: on_amp(amp)
                except Exception: pass
            
            # Start timeout logic
            if not had_speech:
                if elapsed >= start_timeout:
                    if peak_amp >= sp_thr * 0.9: had_speech = True
                    else: break 
                continue 
                
            # Stop logic
            if elapsed >= min_duration and quiet_t >= silence_duration:
                break
            if elapsed >= max_duration:
                break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(samplerate)
        wf.writeframes(b''.join(frames))
        
    return filename, had_speech

# ── Signal processing & Embeddings ──────────────────────────────────────────

def trim_wav_silence(filename: str, threshold=0.02):
    """Trim silence from start and end."""
    try:
        data, sr = sf.read(filename)
        if hasattr(data, "ndim") and data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        amp = np.abs(data)
        idx = np.where(amp > threshold)[0]
        if idx.size == 0: return
        pad = int(0.05 * sr)
        start = max(int(idx[0]) - pad, 0)
        end = min(int(idx[-1]) + pad, len(data))
        if end > start:
            sf.write(filename, data[start:end], sr)
    except Exception as e:
        logger.error(f"Trim failed: {e}")

def compute_embedding(filename: str) -> np.ndarray:
    """Compute speaker embedding (dummy/simple mfcc stats)."""
    try:
        x, sr = sf.read(filename)
        if x.ndim > 1: x = x.mean(axis=1)
        x = x.astype(np.float32)
        
        # Simple stats as embedding (mean/std of magnitude spectrum)
        spec = np.abs(np.fft.rfft(x, n=512))
        mu = spec.mean()
        sigma = spec.std()
        
        # Real implementation would use MFCC/Mel Filterbank
        # Placeholder for 80-dim embedding
        return np.random.randn(80).astype(np.float32) 
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return np.zeros(80, dtype=np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

# ── Liveness check ──────────────────────────────────────────────────────────

def liveness_verify(threshold=0.88) -> bool:
    """Compare a new sample against the owner reference."""
    if not os.path.exists(OWNER_EMBED_PATH):
        return False

    # Note: caller should handle the recording logic or prompt here.
    # This is the verification core.
    try:
        live_embed = compute_embedding(LIVENESS_WAV)
        owner_embed = np.load(OWNER_EMBED_PATH)
        similarity = cosine_similarity(live_embed, owner_embed)
        logger.info(f"Liveness similarity: {similarity:.4f}")
        return similarity >= threshold
    except Exception as e:
        logger.error(f"Liveness verify error: {e}")
        return False
