<<<<<<< HEAD
# sentinel_ai_fixed.py
import os
import time
import wave
import threading
import webbrowser
import subprocess
from queue import Queue
from dotenv import load_dotenv

# audio / VAD / embeddings
import pvporcupine
import pyaudio
import numpy as np
import soundfile as sf

# TTS & STT
import pyttsx3
import speech_recognition as sr
import pythoncom
import httpx
import shutil
import winsound
import requests
import json

# Speaker embedding utilities
# Silero VAD utils
 

# GUI
import tkinter as tk

load_dotenv()

# ---------- Configuration ----------
ACCESS_KEY = os.getenv("PVPORCUPINE_PRIVATE_KEY")  # picovoice key
PORCUPINE_KEYWORD_PATH = "./assets/sounds/Hey-robert_en_windows_v3_0_0.ppn"
REFERENCE_WAV = "reference.wav"
OWNER_EMBED_PATH = "owner_embed.npy"
COMMAND_WAV = "command.wav"
LIVENESS_WAV = "liveness.wav"

# status queue for GUI updates
status_queue = Queue()

agent_active = False
pending_step = None
next_agent_capture_time = 0

def start_agent():
    # Turn on the step-by-step helper. It will wait for steps and ask for confirmation.
    global agent_active, pending_step
    agent_active = True
    pending_step = None
    status_queue.put("Agent active. Describe the first step.")
    speak("Agent started. Describe the first step.")

def stop_agent():
    # Turn off the step-by-step helper.
    global agent_active, pending_step
    agent_active = False
    pending_step = None
    status_queue.put("Agent stopped.")
    speak("Agent stopped.")

def agent_handle(command):
    # Agent understands basic templates:
    # "install X" → runs winget install X
    # "execute Y" → runs a shell command Y
    # "confirm"   → runs the pending step
    global pending_step
    cmd = (command or "").strip()
    if cmd.startswith("install "):
        app = cmd.split("install ", 1)[1].strip()
        pending_step = ("winget", ["install", "--silent", app])
        status_queue.put(f"Pending install: {app}. Say 'confirm' to proceed.")
        speak(f"Ready to install {app}. Say confirm to proceed.")
        return
    if cmd.startswith("execute "):
        raw = cmd.split("execute ", 1)[1].strip()
        pending_step = ("cmd", raw)
        status_queue.put("Pending command. Say 'confirm' to run.")
        speak("Pending command. Say confirm to run.")
        return
    if "confirm" in cmd or "yes" in cmd:
        if not pending_step:
            speak("No pending step.")
            return
        kind, payload = pending_step
        try:
            if kind == "winget":
                subprocess.Popen(["winget"] + payload)
            else:
                subprocess.Popen(payload, shell=True)
            status_queue.put("Step executed. Next step?")
            speak("Step executed. What is the next step?")
        except Exception:
            status_queue.put("Step execution failed.")
            speak("Step execution failed.")
        finally:
            pending_step = None
        return
    speak("Provide a step or say confirm.")

def gemini_generate(prompt):
    # Simple helper that asks Gemini (Google AI) for a response.
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return "Gemini key missing."
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + key
    body = {"contents": [{"parts": [{"text": str(prompt)}]}]}
    try:
        with httpx.Client(timeout=30) as client:
            r = client.post(url, json=body)
            j = r.json()
        return j.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "") or ""
    except Exception:
        return "Gemini error."

def _cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _mel_filterbank(n_fft, sr, n_mels=40, fmin=300.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(m):
        return 700.0 * (10.0**(m / 2595.0) - 1.0)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fbanks = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]
        if center > left:
            fbanks[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fbanks[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fbanks

def compute_embedding(filepath):
    x, sr = sf.read(filepath)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if len(x) < sr // 2:
        pad = sr // 2 - len(x)
        x = np.pad(x, (0, pad))
    x[1:] = x[1:] - 0.97 * x[:-1]
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    n_fft = 512 if sr <= 22050 else 1024
    fb = _mel_filterbank(n_fft, sr, n_mels=40)
    frames = []
    for start in range(0, len(x) - frame_len + 1, hop):
        frame = x[start:start + frame_len]
        frame = frame * np.hamming(frame_len)
        spec = np.fft.rfft(frame, n=n_fft)
        ps = (np.abs(spec) ** 2)
        mel = fb.dot(ps[:fb.shape[1]])
        mel = np.log(mel + 1e-10)
        frames.append(mel)
    if not frames:
        return np.zeros(fb.shape[0] * 2, dtype=np.float32)
    M = np.vstack(frames)
    mu = M.mean(axis=0)
    sigma = M.std(axis=0)
    emb = np.concatenate([mu, sigma]).astype(np.float32)
    return emb

# Silero VAD model (loads on first use)
 

# ---------- Utilities ----------

# Thread-safe TTS (non-blocking)
_tts_lock = threading.Lock()
_tts_disabled = False
def speak(text):
    global _tts_disabled
    def _beep():
        try:
            winsound.Beep(600, 200)
        except Exception:
            pass
    # Check disk space before attempting TTS
    try:
        free_bytes = shutil.disk_usage(os.path.dirname(__file__)).free
        if free_bytes < 10 * 1024 * 1024:
            _tts_disabled = True
    except Exception:
        pass
    if _tts_disabled:
        threading.Thread(target=_beep, daemon=True).start()
        return
    def _s():
        try:
            with _tts_lock:
                pythoncom.CoInitialize()
                engine = pyttsx3.init()
                engine.say(str(text))
                engine.runAndWait()
                pythoncom.CoUninitialize()
        except OSError:
            _tts_disabled = True
            _beep()
        except Exception as e:
            _beep()
    threading.Thread(target=_s, daemon=True).start()

# Record a wav using PyAudio for a fixed duration
def record_wav(filename, duration=3, samplerate=16000, channels=1, frames_per_buffer=1024):
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(format=pyaudio.paInt16,
                         channels=channels,
                         rate=samplerate,
                         input=True,
                         frames_per_buffer=frames_per_buffer)
    except Exception as e:
        pa.terminate()
        raise

    frames = []
    num_frames = int(samplerate / frames_per_buffer * duration)
    for _ in range(num_frames):
        data = stream.read(frames_per_buffer, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # write WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(samplerate)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"[Saved] {filename}")
    return filename

def trim_wav_silence(filename, threshold=0.02):
    # This cuts off the quiet parts at the start and end of a sound file
    # so we keep the important speaking part.
    data, sr = sf.read(filename)
    if hasattr(data, "ndim") and data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    amp = np.abs(data)
    idx = np.where(amp > threshold)[0]
    if idx.size == 0:
        return
    pad = int(0.05 * sr)
    start = max(int(idx[0]) - pad, 0)
    end = min(int(idx[-1]) + pad, len(data))
    if end <= start:
        return
    sf.write(filename, data[start:end], sr)

# Liveness registration / check (Resemblyzer embeddings)
def register_reference_if_missing():
    # If we don't have the owner's voice saved yet,
    # we record a short sample and create its fingerprint.
    if not os.path.exists(OWNER_EMBED_PATH):
        speak("No registered voice found. Please say the passphrase after the beep.")
        time.sleep(0.6)
        print("[Recording reference voice]")
        record_wav(REFERENCE_WAV, duration=3)
        owner_embed = compute_embedding(REFERENCE_WAV)
        np.save(OWNER_EMBED_PATH, owner_embed)
        speak("Voice registered successfully.")
        status_queue.put("Voice registered.")
        return True
    return True

def manual_register_voice():
    # Button in the screen: lets you re-record the owner's voice.
    """Triggered by GUI button: re-record owner reference voice."""
    try:
        speak("Please say your reference passphrase after the beep.")
        time.sleep(0.6)

        status_queue.put("Recording new reference voice...")
        print("[Manual registration] Recording new reference...")

        record_wav(REFERENCE_WAV, duration=3)

        owner_embed = compute_embedding(REFERENCE_WAV)
        np.save(OWNER_EMBED_PATH, owner_embed)

        speak("Voice registration updated successfully.")
        status_queue.put("Voice registration updated successfully.")

        print("[Manual registration] Updated owner_embed.npy")
    except Exception as e:
        speak("Voice registration failed.")
        status_queue.put(f"Registration error: {e}")
        print("[Manual registration error]", e)

def liveness_check(threshold=0.80):
    # Checks quickly if the new voice sample looks like the owner's
    # by comparing their fingerprints.
    # Ensure reference exists
    if not os.path.exists(OWNER_EMBED_PATH):
        register_reference_if_missing()

    speak("Please repeat the passphrase after the beep.")
    time.sleep(0.5)
    print("[Recording liveness sample]")
    record_wav(LIVENESS_WAV, duration=3)
    try:
        live_embed = compute_embedding(LIVENESS_WAV)
        owner_embed = np.load(OWNER_EMBED_PATH)
        similarity = _cosine(live_embed, owner_embed)
        print(f"[Liveness Similarity Score]: {similarity:.4f}")
        status_queue.put(f"Liveness score: {similarity:.3f}")
        return similarity >= threshold
    except Exception as e:
        print("[Liveness error]", e)
        return False

# Transcribe a WAV file using SpeechRecognition (Google)
def transcribe_wav(filename):
    # Turns a voice recording into text using Google's free speech tool.
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        print("[Transcribed]:", text)
        return text.lower()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print("[Speech API error]", e)
        return ""

# Interpret command using OpenAI (simple wrapper)
def interpret_command(command):
    # If you say: "ask gemini something", we ask Gemini (Google AI).
    # Otherwise we ask OpenAI to summarize the command.
    c = (command or "").lower()
    if c.startswith("ask gemini") or c.startswith("use gemini"):
        q = command.split(" ", 2)
        prompt = q[2] if len(q) > 2 else command
        return gemini_generate(prompt)
    from openai import OpenAI
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return "No OpenAI key configured."
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,)
    prompt = f"You are a helpful desktop assistant. Convert the command into a short action summary. Command: {command}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are a helpful desktop assistant."},
                {"role":"user","content":prompt}
            ],
            max_tokens=120
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        import openai
        openai.api_key = key
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role":"system","content":"You are a helpful desktop assistant."},
                {"role":"user","content":prompt}
            ],
            max_tokens=120
        )
        return resp.choices[0].message.content.strip()

# Execute a handful of commands (keeps your original behaviors)
def execute_command(command):
    # This is the action center. It reads the command text and does the matching thing.
    command = (command or "").lower()
    print("[Execute] ", command)
    global agent_active
    if agent_active:
        agent_handle(command)  # When agent is active, treat every sentence as a step.
        return
    if command in ("start", "start agent", "agent start", "enable agent"):
        start_agent()  # Turn on agent mode
        return
    if command in ("stop", "stop agent", "agent stop", "disable agent"):
        stop_agent()   # Turn off agent mode
        return
    if "open browser" in command:
        webbrowser.open("https://www.google.com")
    elif "open gmail" in command:
        webbrowser.open("https://mail.google.com")
=======

import os
import subprocess
import webbrowser
import pyttsx3
import speech_recognition as sr
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import sounddevice as sd
import soundfile as sf
import time
import tkinter as tk
from tkinter import messagebox
import threading
import pvporcupine
import struct
import pyaudio
import openai
from queue import Queue
from dotenv import load_dotenv, dotenv_values

load_dotenv() 


status_queue = Queue()

# Set your OpenAI API key
openai.api_key = os.getenv("OPEN_AI_API_KEY")  # Or replace with your key string

encoder = VoiceEncoder()

# Initialize TTS engine
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Record audio to file
def record_audio_to_file(filename="liveness.wav", duration=3):
    samplerate = 16000
    print(f"[Recording] Saving to {filename} for {duration} seconds...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, recording, samplerate)
    print("[Recording Finished]")
    print(f"[Saved] {filename}")
    return filename


def liveness_check():
    global session_valid_until
    encoder = VoiceEncoder()

    # First-time setup or reference loading
    reference_path = "testcode.wav"
    owner_embed_path = "owner_embed.npy"

    if not os.path.exists(owner_embed_path):
        if not os.path.exists(reference_path):
            speak("Reference voice not found. Please register your voice.")
            record_audio_to_file(reference_path, duration=3)
        wav = preprocess_wav(reference_path)
        owner_embed = encoder.embed_utterance(wav)
        np.save(owner_embed_path, owner_embed)
        speak("Voice registered successfully.")
        return True

    # Prompt and record live input
    speak("Please repeat the phrase: I am the master.")
    record_audio_to_file("liveness.wav", duration=3)

    try:
        live_wav = preprocess_wav("liveness.wav")
        live_embed = encoder.embed_utterance(live_wav)
        owner_embed = np.load(owner_embed_path)

        similarity = np.inner(live_embed, owner_embed)
        print(f"[Liveness Similarity Score]: {similarity:.4f}")

        return similarity > 0.40
    except Exception as e:
        print(f"[Error in liveness check]: {e}")
        return False

# Transcribe command
def get_command():
    recognizer = sr.Recognizer()
    with sr.AudioFile("liveness.wav") as source:
        audio = recognizer.record(source)
    try:
        command = recognizer.recognize_google(audio)
        print("Google Speech Recognition thinks you said " + recognizer.recognize_google(audio))
        print("command:", command)
        return command.lower()
    except sr.UnknownValueError:
        print("Google Speech Recognition thinks you said " + recognizer.recognize_google(audio))
        speak("Sorry, I did not understand.")
        return ""
    except sr.RequestError:
        print("Google Speech Recognition thinks you said " + recognizer.recognize_google(audio))
        speak("Sorry, my speech service is currently unavailable.")
        return ""

# Use OpenAI to interpret command and generate a response or action

def interpret_command(command):
    print("command:", command)
    prompt = f"You are a smart desktop assistant. Interpret the user's voice command and suggest a Python function call.\nCommand: {command}\nRespond with the best matching action."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for executing desktop tasks."},
                {"role": "user", "content": prompt}
            ]
        )
        action = response.choices[0].message.content.strip()
        return action
    except Exception as e:
        speak("I encountered an error accessing my intelligence service.")
        return ""

# Execute interpreted action
def execute_command(command):
    
    if "open browser" in command:
        webbrowser.open("https://www.google.com")
    elif "open gmail" in command:
        webbrowser.open("https://www.google.com/gmail")
>>>>>>> b842af7c098ee245d7c2c3eedf6d4375cf1c8ffd
    elif "open youtube" in command:
        webbrowser.open("https://www.youtube.com")
    elif "open facebook" in command:
        webbrowser.open("https://www.facebook.com")
    elif "open github" in command:
        webbrowser.open("https://www.github.com")
<<<<<<< HEAD
    elif "open settings" in command:
        os.system("start ms-settings:")
    elif "open task manager" in command or "task manager" in command:
        subprocess.Popen(["taskmgr"])
    elif "open powershell" in command:
        subprocess.Popen(["powershell.exe"])
    elif "open vscode" in command or "open vs code" in command:
        paths = [
            os.path.expandvars(r"%LocalAppData%\Programs\Microsoft VS Code\Code.exe"),
            r"C:\\Program Files\\Microsoft VS Code\\Code.exe"
        ]
        exe = next((p for p in paths if os.path.exists(p)), None)
        if exe:
            subprocess.Popen([exe])
        else:
            try:
                subprocess.Popen(["code"])
            except Exception:
                speak("VS Code not found.")
    elif "system info" in command:
        try:
            out = subprocess.check_output(["systeminfo"], shell=True, text=True, timeout=20)
            status_queue.put("System info collected.")
            print(out[:1000])
            speak("System information collected.")
        except Exception:
            status_queue.put("System info failed.")
    elif "ip address" in command or "show ip" in command:
        try:
            out = subprocess.check_output(["ipconfig"], shell=True, text=True, timeout=15)
            lines = [l.strip() for l in out.splitlines() if "IPv4" in l]
            msg = "; ".join(lines) or "No IPv4 found."
            status_queue.put(msg)
            speak("IP information updated.")
        except Exception:
            status_queue.put("IP check failed.")
    elif "battery report" in command:
        try:
            report = os.path.join(os.path.dirname(__file__), "battery-report.html")
            subprocess.check_call(["powercfg", "/batteryreport", "/output", report], shell=True)
            os.startfile(report)
            status_queue.put("Battery report opened.")
            speak("Battery report opened.")
        except Exception:
            status_queue.put("Battery report failed.")
    elif command.startswith("ask gemini") or command.startswith("use gemini"):
        q = command.split(" ", 2)
        prompt = q[2] if len(q) > 2 else command
        resp = gemini_generate(prompt)
        speak(resp)
    elif "start agent" in command or command.startswith("agent"):
        start_agent()
    elif "stop agent" in command:
        stop_agent()
    elif "open gpt" in command:
        try:
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            import speech_recognition as sr
            import time
        except Exception:
            speak("Web automation unavailable.")
            return
=======
    elif "open gpt" in command:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        import speech_recognition as sr
        import time
>>>>>>> b842af7c098ee245d7c2c3eedf6d4375cf1c8ffd

        # Set up Chrome options
        options = Options()
        options.add_argument("--start-maximized")  # Open browser maximized

        # Start driver
        service = Service("chromedriver")  # Adjust if not in PATH
        driver = webdriver.Chrome(service=service, options=options)
        driver.get("https://chatgpt.com/?model=auto")

        # Wait for page load
        wait = WebDriverWait(driver, 30)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "textarea")))

        # Type voice input
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Speak your message to ChatGPT:")
            audio = recognizer.listen(source)

        try:
            query = recognizer.recognize_google(audio)
            print(f"You said: {query}")
            textarea = driver.find_element(By.TAG_NAME, "textarea")
            textarea.send_keys(query)
            time.sleep(1)

            # Click the submit button using its ID
            send_button = driver.find_element(By.ID, "composer-submit-button")
            send_button.click()
            print("Query sent to ChatGPT.")

        except sr.UnknownValueError:
            print("Sorry, could not understand your voice.")
        except Exception as e:
            print("Error:", e)


    elif "shutdown" in command:
<<<<<<< HEAD
        speak("Shutting down. Goodbye.")
        os.system("shutdown /s /t 1")
    elif "restart" in command:
        os.system("shutdown /r /t 1")
    elif "open notepad" in command or "notepad" in command:
        subprocess.Popen(["notepad.exe"])
    elif "open chrome" in command or "chrome" in command:
        query = command.replace("open chrome", "").replace("search", "").strip() or "python programming"
        webbrowser.open(f"https://www.google.com/search?q={query.replace(' ', '+')}")
        speak(f"Searching Chrome for {query}")
    else:
        # fallback interpreter + speak
        resp = interpret_command(command)
        speak(resp)

# ---------- Wake-word listener (runs in background thread) ----------
def listen_for_wake_word_loop(session_duration=3600):
    # Background loop:
    # 1) Wait for wake word (or timed agent capture)
    # 2) Check voice liveness (if needed)
    # 3) Record short command and execute it
    """Listens for Porcupine wake word and then handles auth + command."""
    session_valid_until = 0
    if not ACCESS_KEY:
        status_queue.put("Porcupine key missing. Wake word disabled.")
        return
    try:
        porcupine = pvporcupine.create(
            access_key=ACCESS_KEY,
            keyword_paths=[PORCUPINE_KEYWORD_PATH] if os.path.exists(PORCUPINE_KEYWORD_PATH) else None,
            keywords=["hey computer"] if not os.path.exists(PORCUPINE_KEYWORD_PATH) else None,
            sensitivities=[0.6]
        )
    except Exception as e:
        print("[Porcupine init error]", e)
        status_queue.put("Porcupine init failed.")
        return

    pa = pyaudio.PyAudio()  # open the microphone for reading
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=porcupine.sample_rate,
                     input=True,
                     frames_per_buffer=porcupine.frame_length)

    status_queue.put("Listening for wake word...")
    print("[Porcupine running]")

    try:
        while True:
            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm_unpacked = np.frombuffer(pcm, dtype=np.int16)
            keyword_index = porcupine.process(pcm_unpacked)
            now_t = time.time()
            should_capture = keyword_index >= 0 or (agent_active and now_t >= next_agent_capture_time)
            if should_capture:
                print("[Wake word detected]")
                status_queue.put("Wake word detected.")
                if not agent_active:
                    speak("Yes Master?")
                now = time.time()
                if now > session_valid_until and not agent_active:
                    status_queue.put("Performing liveness check...")
                    if not liveness_check():
                        speak("Access denied. Voice does not match.")
                        status_queue.put("Liveness failed.")
                        continue
                    else:
                        session_valid_until = time.time() + session_duration
                        status_queue.put("Liveness passed. Session active.")
                        speak("Liveness confirmed. Please speak your command after the beep.")

                # Record user command
                time.sleep(0.25)
                status_queue.put("Recording command...")
                record_wav(COMMAND_WAV, duration=4)
                status_queue.put("Processing command...")

                try:
                    trim_wav_silence(COMMAND_WAV)
                except Exception as e:
                    print("[Trim error]", e)

                # transcribe & execute
                cmd_text = transcribe_wav(COMMAND_WAV)
                if not cmd_text:
                    speak("Sorry, I couldn't understand the command.")
                    status_queue.put("Transcription empty.")
                else:
                    status_queue.put(f"Command: {cmd_text}")
                    execute_command(cmd_text)

                if agent_active:
                    next_agent_capture_time = time.time() + 6  # listen again soon (in 6s)

    except Exception as e:
        print("[Wake loop error]", e)
        status_queue.put("Error in wake listener.")
    finally:
        try:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            porcupine.delete()
        except Exception:
            pass

# ---------- GUI ----------
# def start_gui():

#     root = tk.Tk()
#     root.title("SentinelAI Dashboard")
#     root.geometry("380x180")

#     status_label = tk.Label(root, text="SentinelAI Running...", font=("Arial", 11))
#     status_label.pack(pady=8)

#     session_label = tk.Label(root, text="Session: expired", font=("Arial", 10))
#     session_label.pack()

#     diag_items = []
#     diag_items.append("OpenAI: OK" if os.getenv("OPEN_AI_API_KEY") else "OpenAI: Missing")
#     diag_items.append("Porcupine: OK" if os.getenv("PVPORCUPINE_PRIVATE_KEY") else "Porcupine: Missing")
#     diag_items.append("Driver: OK" if os.path.exists(os.path.join(os.path.dirname(__file__), "chromedriver-win64", "chromedriver.exe")) else "Driver: Missing")
#     diag_label = tk.Label(root, text=" | ".join(diag_items), font=("Arial", 9))
#     diag_label.pack(pady=6)

#     def update_ui():
#         # drain status queue
#         while not status_queue.empty():
#             msg = status_queue.get_nowait()
#             status_label.config(text=msg)
#         root.after(800, update_ui)

#     update_ui()
#     root.mainloop()

def start_gui():
    # Builds a small window with useful buttons and status messages.
    root = tk.Tk()
    root.title("SentinelAI Dashboard")
    root.geometry("420x240")

    status_label = tk.Label(root, text="SentinelAI Running...", font=("Arial", 11))
    status_label.pack(pady=8)

    session_label = tk.Label(root, text="Session: expired", font=("Arial", 10))
    session_label.pack()

    # Diagnostics row: shows whether keys and driver are present
    diag_items = []
    diag_items.append("OpenAI: OK" if os.getenv("OPEN_AI_API_KEY") else "OpenAI: Missing")
    diag_items.append("Porcupine: OK" if os.getenv("PVPORCUPINE_PRIVATE_KEY") else "Porcupine: Missing")
    driver_path = os.path.join(os.path.dirname(__file__), "chromedriver-win64", "chromedriver.exe")
    diag_items.append("Driver: OK" if os.path.exists(driver_path) else "Driver: Missing")
    
    diag_label = tk.Label(root, text=" | ".join(diag_items), font=("Arial", 9))
    diag_label.pack(pady=6)

    register_btn = tk.Button(
        root,
        text="Register Voice",
        font=("Arial", 11),
        padx=10,
        pady=4,
        command=manual_register_voice
    )
    register_btn.pack(pady=8)

    def manual_run_command():
        # Click this to record a command without saying the wake word.
        status_queue.put("Recording command...")
        record_wav(COMMAND_WAV, duration=4)
        try:
            trim_wav_silence(COMMAND_WAV)
        except Exception as e:
            print("[Trim error]", e)
        text = transcribe_wav(COMMAND_WAV)
        if not text:
            status_queue.put("Transcription empty.")
            speak("Sorry, I couldn't understand the command.")
        else:
            status_queue.put(f"Command: {text}")
            execute_command(text)

    cmd_btn = tk.Button(
        root,
        text="Record & Execute Command",
        font=("Arial", 11),
        padx=10,
        pady=4,
        command=manual_run_command
    )
    cmd_btn.pack(pady=6)

    agent_start_btn = tk.Button(
        root,
        text="Start Agent",
        font=("Arial", 11),
        padx=10,
        pady=4,
        command=start_agent
    )
    agent_start_btn.pack(pady=4)

    agent_stop_btn = tk.Button(
        root,
        text="Stop Agent",
        font=("Arial", 11),
        padx=10,
        pady=4,
        command=stop_agent
    )
    agent_stop_btn.pack(pady=4)

    # Quit button
    quit_button = tk.Button(root, text="Quit", padx=8, pady=4, command=root.destroy)
    quit_button.pack(pady=8)

    # Update UI loop
    def update_ui():
        # Every little while, take messages from the queue and show them on screen
        while not status_queue.empty():
            msg = status_queue.get_nowait()
            status_label.config(text=msg)
        root.after(700, update_ui)

    update_ui()
    root.mainloop()


# ---------- Main ----------
def main():
    # Program start:
    # 1) Make sure we have the owner's voice saved
    # 2) Start listening in the background
    # 3) Show the small dashboard
    # Ensure owner reference exists (register if missing)
    register_reference_if_missing()

    # Start listener thread
    t = threading.Thread(target=listen_for_wake_word_loop, daemon=True)
    t.start()

    # Start GUI (blocks main thread)
    start_gui()

if __name__ == "__main__":
    main()  # run the program when we start this file
=======
        os.system("shutdown /s /t 1")
    elif "restart" in command:
        os.system("shutdown /r /t 1")
    elif "sleep" in command:
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
    elif "lock computer" in command:
        os.system("rundll32.exe user32.dll,LockWorkStation")
    elif "open notepad" in command or "notepad" in command:
        subprocess.Popen("notepad.exe")
    elif "open chrome" in command or "chrome" in command:
        import pyttsx3
        query = command.replace("open chrome", "").replace("search", "").strip() or "python programming"
    
        # Register Chrome path if needed
        webbrowser.register(
            "chrome",
            None,
            webbrowser.BackgroundBrowser("C://Program Files//Google//Chrome//Application//chrome.exe")
        )

        # Prepare search URL
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"

        # Open in Chrome
        webbrowser.get("chrome").open(search_url)

        # Speak back confirmation
        engine = pyttsx3.init()
        engine.say(f"Searching Google for {query}")
        engine.runAndWait()

    elif "open calculator" in command:
        subprocess.Popen("calc.exe")
    elif "open command prompt" in command or "open cmd" in command:
        subprocess.Popen("cmd.exe")
    elif "open paint" in command:
        subprocess.Popen("mspaint.exe")
    elif "open explorer" in command:
        subprocess.Popen("explorer.exe")
    elif "open downloads" in command:
        downloads = os.path.join(os.path.expanduser("~"), "Downloads")
        os.startfile(downloads)
    elif "open documents" in command:
        documents = os.path.join(os.path.expanduser("~"), "Documents")
        os.startfile(documents)
    elif "open desktop" in command:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        os.startfile(desktop)
    elif "destroy yourself" in command or "quit app" in command or "close assistant" in command:
        speak("Goodbye, Master. Shutting down.")
        os._exit(0)
    elif "wake up" in command or "unlock" in command:
        import pyautogui
        pyautogui.moveRel(0, 1)  # Tiny mouse movement to wake screen
        pyautogui.press("shift")  # Simulate shift key to wake lock screen
        speak("Screen is awake, Master. Please enter your code.")
    elif "unlock with" in command:
        passcode = command.split("unlock with")[-1].strip()
        if passcode == "1234":
            global session_valid_until
            session_valid_until = time.time() + 3600  # 1 hour session
            speak("Access granted. Welcome back, Master.")
            status_queue.put("Unlocked via passcode. Session renewed.")
        else:
            speak("Incorrect passcode.")
            status_queue.put("Failed unlock attempt with wrong passcode.")

    elif "type" in command:
        import pyautogui
        text = command.split("type")[-1].strip()
        pyautogui.write(text, interval=0.05)
    else:
        interpreted = interpret_command(command)
        speak(interpreted)

# Wake word listener
session_valid_until = 0

def listen_for_wake_word():
    global session_valid_until

    try:
        porcupine = pvporcupine.create(
            access_key=os.getenv("PVPORCUPINE_PRIVATE_KEY"),
            keyword_paths=["./assets/sounds/Hey-robert_en_windows_v3_0_0.ppn"],
            sensitivities=[0.6]
        )
        print("[Porcupine Wake Word Engine Initialized]")

        pa = pyaudio.PyAudio()
        print("[PyAudio Initialized]")
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=porcupine.sample_rate,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )

        status_queue.put("Listening for wake word...")

        while True:
            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcm)

            if keyword_index >= 0:
                print("[Wake Word Detected]")
                status_queue.put("Wake word detected! Verifying voice...")
                speak("Yes Master?")
                current_time = time.time()

                if current_time > session_valid_until:
                    if liveness_check():
                        print("[Liveness Confirmed]")
                        status_queue.put("Liveness check passed!")
                        speak("Liveness check passed. Please give your command.")
                        session_valid_until = time.time() + 3600  # 1 hour
                    else:
                        print("[Liveness Failed]")
                        status_queue.put("Access denied: voice mismatch.")
                        speak("Access denied. Voice does not match.")
                        continue

                record_audio_to_file()
                print("[Recording Command]")
                status_queue.put("Command recorded. Recognizing...")

                command = get_command()
                print(f"[Command]: {command}")
                status_queue.put(f"Executing: {command}")

                if command:
                    execute_command(command)

    except Exception as e:
        print(f"[Error in wake word listener]: {e}")
        status_queue.put("Error in microphone or wake word engine.")

    finally:
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            if pa is not None:
                pa.terminate()
            if porcupine is not None:
                porcupine.delete()
        except:
            pass




# GUI dashboard

def start_gui():
    def update_status():
        while not status_queue.empty():
            msg = status_queue.get_nowait()
            status_label.config(text=msg)
        root.after(1000, update_status)
     
    def update_session_time():
        current_time = time.time()
        if session_valid_until > current_time:
            remaining = int(session_valid_until - current_time)
            mins, secs = divmod(remaining, 60)
            session_label.config(text=f"Session: {mins:02d}:{secs:02d} remaining")
        else:
            session_label.config(text="Session expired. Awaiting liveness.")


    root = tk.Tk()
    root.title("SentinelAI Dashboard")
    root.geometry("300x150")

    status_label = tk.Label(root, text="SentinelAI Running...", font=("Arial", 12))
    status_label.pack(pady=10)

    session_label = tk.Label(root, text="", font=("Arial", 10))
    session_label.pack(pady=5)

    quit_button = tk.Button(root, text="Quit", command=root.destroy)
    quit_button.pack(pady=10)
    
    root.after(1000, update_status)
    root.mainloop()

# Main app entry point
def main():
    threading.Thread(target=listen_for_wake_word, daemon=True).start()
    start_gui()

if __name__ == "__main__":
    main()
>>>>>>> b842af7c098ee245d7c2c3eedf6d4375cf1c8ffd
