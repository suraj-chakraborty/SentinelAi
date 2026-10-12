# SentinelAI — Complete Voice & Tactical Command Reference

This is the comprehensive, authoritative guide to all voice directives, shortcuts, and capabilities available in **SentinelAI**.

---

## ⚡ Quick Start & Global Shortcuts

| Action | Keyboard / Mouse Shortcut | Command |
| :--- | :--- | :--- |
| **Summon Floating Companion** | <kbd>Win</kbd> + <kbd>Space</kbd> or <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>Space</kbd> or <kbd>Alt</kbd> + <kbd>J</kbd> | `"launch companion"`, `"open widget"` |
| **Instant Voice Barge-In** | <kbd>Spacebar</kbd> (in web HUD or on companion) | `"stop speaking"` / Speak over Sentinel |
| **Hands-Free Streaming (VAD)** | Header toggle in Web UI (`http://127.0.0.1:5000`) | Native microphone speech onset |
| **Wake Word Activation** | Acoustic Voice (`"Hey Robert"` / `"Hey Sentinel"`) | Say wake word, then speak directive |

---

## 1. 🖥️ Workstation, Power & Window Management

### System Power
- **`shutdown`** | **`shutdown computer`** | **`power off pc`** — Initiates graceful 60-second system shutdown.
- **`restart`** | **`reboot computer`** | **`restart system`** — Reboots the Windows workstation.
- **`lock computer`** | **`lock pc`** | **`lock screen`** — Immediately locks your desktop.
- **`hibernate`** — Saves state to disk and hibernates the computer.
- **`sleep`** | **`suspend`** — Puts computer into low-power sleep state.
- **`deactivate`** | **`exit sentinel`** | **`quit sentinel`** — Shuts down SentinelAI assistant safely.

### Window & Screen Automation
- **`snap left`** | **`tile left`** — Snaps active foreground window to left half of screen.
- **`snap right`** | **`tile right`** — Snaps active foreground window to right half of screen.
- **`center window`** — Centers foreground window on active display.
- **`minimize window`** | **`maximize window`** | **`close window`** — Standard window state toggles.
- **`organize downloads`** — Automatically cleans and sorts `~/Downloads` into categories.

### Hardware & Display Control
- **`set brightness to <X>%`** | **`brightness <X>%`** — Adjusts monitor brightness (e.g., `"set brightness to 75%"`).
- **`brightness up`** | **`increase brightness`** — Increases screen brightness by 10%.
- **`brightness down`** | **`decrease brightness`** — Lowers screen brightness by 10%.
- **`set volume to <X>%`** — Sets system master volume level.
- **`volume up`** | **`volume down`** | **`louder`** | **`quieter`** — Adjusts volume incrementally.
- **`mute`** | **`unmute`** | **`sound on`** — Toggles audio output mute state.

---

## 2. 📱 Applications, Browsing & Web Automation

### Application Launching & Process Control
- **`open <app name>`** | **`launch <app name>`** — Launches software (e.g., `"open chrome"`, `"launch vscode"`, `"run notepad"`, `"open spotify"`).
- **`close <app name>`** | **`kill <app name>`** | **`terminate <app name>`** — Closes application process.
- **`open browser`** | **`open gmail`** | **`open youtube`** | **`open github`** — Direct shortcuts to popular portals.

### Web Browsing & Autonomous Scraping
- **`go to <url>`** | **`navigate to <url>`** | **`open <url>`** — Directly navigates default browser (e.g., `"go to github.com"`).
- **`search for <query>`** | **`google <query>`** — Searches Google without leaving context.
- **`scrape <website>`** | **`compare prices`** | **`check flights`** — Autonomous Playwright browser flow.
- **`computer use <task>`** — Hands-free visual cursor & keyboard automation.

---

## 3. 📝 Productivity, Notes, Clipboard & Encrypted Vault

### Quick Notes
- **`take a note: <text>`** | **`remember note <text>`** — Stores a note into local persistent store.
- **`show notes`** | **`list notes`** — Displays or speaks recent saved notes.
- **`forget note <number>`** — Removes note by index.

### Encrypted Password Vault
- **`password for <service> is <value>`** — Encrypts and saves secret credentials safely.
- **`variable <name> is <value>`** — Stores secure configuration tokens.
- **`password for <service> give me`** | **`show password for <service>`** — Decrypts and retrieves credential.

### System Clipboard
- **`read clipboard`** | **`what did i copy`** | **`paste`** — Reads active clipboard buffer.
- **`copy to clipboard <text>`** — Injects text into system clipboard.

---

## 4. ⏰ Timers, Alarms, Reminders & Schedule

### Timers & Alarms
- **`set a timer for <X> minutes`** | **`timer for <X>`** — Starts a countdown timer with alarm audio.
- **`set an alarm for <time>`** | **`wake me at <time>`** — Arms alarm for target hour (e.g., `"wake me at 7 AM"`).

### Reminders & Calendar
- **`remind me at <time> to <task>`** — Sets timed reminder notifications.
- **`my schedule`** | **`what's on my calendar today`** — Retrieves today's calendar agenda.
- **`schedule meeting`** | **`add event`** — Books event via Google Calendar integration.

---

## 5. 📁 File Intelligence & Local Search

- **`find file <filename/pattern>`** — Scans local directories for matching files.
- **`read file <filepath>`** — Reads and summarizes contents of target file.
- **`create file <filepath>`** | **`write file <filepath>`** — Creates new file on disk.
- **`clean downloads`** | **`find large files`** — Identifies storage hogs and clears temporary files.

---

## 6. 🧮 Fast Smart Responses (Local Q&A, Math & Translation)

- **`calculate <expression>`** | **`what is <X> * <Y>`** — Instant local math computation.
- **`what is <topic>`** | **`explain <topic>`** | **`define <concept>`** — Concise factual summary without opening apps.
- **`weather`** | **`temperature outside`** | **`forecast`** — Current weather & meteorological updates.
- **`news`** | **`top headlines`** | **`what's happening today`** — Curated news digest.
- **`translate <text> to <language>`** | **`how do you say <text> in <language>`** — Multi-lingual neural translation (Spanish, French, German, Japanese, Chinese, Korean, Italian, Hindi, Russian, Arabic).

---

## 7. 🧠 Memory & Episodic Recall (ChromaDB + SQLite)

- **`remember this <fact>`** | **`add to memory <fact>`** — Embeds knowledge into ChromaDB vector store.
- **`what do you know about <topic>`** | **`search memory <topic>`** — Semantic RAG retrieval.
- **`what did we do yesterday?`** | **`previous session`** — Episodic log analysis of past interactions.
- **`recent commands`** | **`command history`** — Reviews recently executed actions.
- **`search screen memory for <topic>`** — Recalls past captured screen frames by OCR content.

---

## 8. 💻 Pair-Programming, Git & Developer Intelligence

### Voice Pair-Programmer (Code Whisperer)
- **`start pair programming`** | **`code whisperer on`** — Activates interactive IDE companion.
- **`explain this code`** | **`walk me through this`** — AST structural breakdown of highlighted code.
- **`generate a function <name>`** | **`create code for <task>`** — Generates typed code from spoken prompt.
- **`debug this error`** | **`rubber duck`** — Interactive reasoning through logic flaws and tracebacks.
- **`end pair programming`** — Concludes session and summarizes fixes.

### Git & IDE Copilot
- **`commit changes`** | **`git commit with message <msg>`** — Stages and commits with intelligent message generation.
- **`git status`** | **`what files changed`** — Summarizes working tree state.
- **`analyze terminal error`** | **`why did my build fail`** — Reads terminal error from clipboard and explains root cause.

---

## 9. 🛡️ Focus Shield, Ergonomics & Meeting Co-Pilot

### Focus Shield & Ergonomics
- **`start focus mode [X minutes]`** | **`deep focus`** — Silences notifications and closes distracting apps.
- **`stop focus mode`** | **`end focus`** — Restores normal notification flow.
- **`posture check`** | **`how long have i worked`** — Ergonomic screen-time and fatigue metrics.

### Meeting Co-Pilot (Surrogate)
- **`start meeting`** | **`join meeting`** | **`listen to meeting`** — Captures system call audio in background.
- **`end meeting`** | **`stop meeting`** — Concludes meeting and extracts action items.
- **`meeting summary`** | **`what did we discuss`** | **`action items`** — Delivers bulleted action matrix.

---

## 10. 🎭 Dynamic Mood Resonance, Flirting & Voice Cloning

### Emotional Personality Modes
- **`flirt with me`** | **`be flirty`** — Playfully charming responses (*"Careful, you might make my GPU blush ~"*).
- **`be angry`** | **`cute anger`** | **`pout`** | **`why are you pouting`** — Tsundere pout reactions (*"Hmph! What did you do to your code this time?!"*).
- **`be playful`** | **`joke around`** — Quick-witted banter and tech humor.
- **`comfort me`** | **`i need comfort`** — Empathic, supportive guidance.
- **`what is your mood`** — Reports active internal mood and acoustic resonance score.
- **`what do you think of <topic>`** — Opinionated stance on tech, architecture, and design.

### Voice Cloning & Profiles
- **`record voice sample`** | **`clone my voice`** — Captures 5-second reference to synthesize personalized agent timbre.
- **`reset agent voice`** — Restores default neural voice profile.
- **`show voice profile`** — Displays active acoustic parameters (pitch offset, rate offset).

---

## 11. 🚀 Frontier JARVIS Capabilities

### 🧠 Nightly Autonomous Swarm & Dream State
- **`run nightly audit`** | **`start nightly swarm`** — 3-agent maintenance cycle (audits git dependencies, crawls ArXiv/GitHub trends, patches errors).
- **`enter dream state`** | **`night mode`** — Background consolidation: Light sleep disk cleanup, Deep dream self-healing, Waking briefing.
- **`wake up`** | **`morning report`** | **`dream status`** — Delivers executive morning briefing.

### ⚡ Native Speech-to-Speech Streaming (Sub-200ms)
- **`start live speech`** | **`fast voice mode`** — Full-duplex WebSocket stream with **Adaptive Acoustic Mirroring** (whispering vs projection).
- **`translate live to [spanish/japanese/german/french]`** — Real-time in-call cloned voice translation for Discord/Zoom/Teams.
- **`stop live speech`** — Returns to standard wake-word mode.

### 🏡 Room-Scale Spatial IoT & Phone Bridge
- **`focus lighting`** | **`sync desk light`** — Bias backlighting and focus room scenes.
- **`cinema mode`** | **`movie mode`** — Low-light media scene profile.
- **`hand off audio to [kitchen/bedroom/office]`** — Seamlessly transfers audio to target smart speaker.
- **`who called me`** | **`missed calls`** — Inspects smartphone incoming call logs.
- **`phone messages`** | **`text messages`** — Summarizes SMS and WhatsApp phone notifications.

### 🖐️ Touchless Air-Gestures & Biometrics
- **`start gesture control`** | **`air gesture on`** — MediaPipe 30 FPS touchless camera gestures (Palm hold, Swipe, Pinch, Fist close, Circle volume).
- **`stop gesture control`** — Halts webcam gesture detection.
- **`start health monitor`** | **`check my vitals`** — Webcam rPPG heart rate, eye blink rate, posture, and wellness score.

### ⚡ NPU/GPU Acceleration & Task Dispatcher
- **`npu status`** | **`gpu status`** — DirectML, CUDA, OpenVINO, ONNX Runtime hardware offloading metrics.
- **`recommend quantization [model]`** — INT4 / INT8 / FP16 VRAM recommendations.
- **`start task dispatcher`** | **`dispatcher status`** — Multi-device DAG worker pool.
- **`mesh devices`** — Inspects connected local, mobile, and cloud worker nodes.

---

## 12. 🌌 The 4 Flagship Cyberpunk Capabilities

### A. 🔮 Real-Time 3D Viseme Lip-Sync (Audio-Reactive Avatar Morphing)
- **Three.js WebGL Hologram**: Live rotating wireframe core with dynamic particle aura in the web HUD.
- **Formant Decomposition**:
  - `Formant 1 (100–400Hz)`: Vertical aperture dilation for vowels `AA` / `OH` (`scale.y` expansion).
  - `Formant 2 (800–2200Hz)`: Horizontal equatorial stretch for vowels `EE` / `IH` (`scale.x` expansion).
  - `High Frequencies (3000–6000Hz)`: Perimeter surface agitation for sibilants `TH` / `SS`.
- **Live HUD Viseme Badge**: Real-time phonetic states: `[VISEME: OH / AA · OPEN]`, `[VISEME: EE · WIDE]`, `[VISEME: TH / SS · SIBILANT]`, `[VISEME: MM · CLOSED]`.

### B. 👁️ Live Screen Vision HUD (Proactive Error Detection & Bounding Boxes)
- **`inspect screen errors`** | Click **`🔍 Inspect Screen Errors`** in Web HUD — Captures monitor viewport via `GET /vision/screen/inspect` and inspects foreground window for exceptions.
- **Glowing Neon Bounding Boxes**: Crimson `#f43f5e` for tracebacks/errors, Cyan `#00f3ff` for active focus elements rendered directly on `<canvas id="screen-hud-canvas">`.
- **`⚡ Auto-Fix Detected Error`** — Automatically prompts Code Whisperer & orchestrator to draft a solution for detected crashes.

### C. 🎙️ Browser-Native Hands-Free Voice Streaming (Sub-300ms S2S)
- **Continuous Voice Activity Detection (VAD)**: Click **`🎙️ Hands-Free Voice`** in header. Web Audio API tracks microphone energy without manual push-to-talk.
- **Sub-300ms Speech Turnaround**: Automatically captures utterances upon speech onset, detects 500ms of silence, and streams base64 audio over WebSocket `/ws`.
- **Instant Full-Duplex Barge-In**: Hit <kbd>Spacebar</kbd> anywhere or speak over Sentinel to immediately interrupt active speech and audio generation.

### D. 🪟 Transparent Floating Desktop Widget (`Win + Space` Companion)
- **Always-on-Top Frameless Overlay**: Sleek dark obsidian glass overlay (`-alpha 0.94`, `#080c14`) with animated 2D holographic radar orb, draggable anywhere on your screens.
- **`launch companion`** | **`open widget`** | **`toggle companion`** — Toggles the desktop overlay.
- **Global Hotkeys Hooked**:
  - <kbd>Win</kbd> + <kbd>Space</kbd>
  - <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>Space</kbd>
  - <kbd>Alt</kbd> + <kbd>J</kbd>
- **Dedicated Controls**: Text command input with Enter execution, dedicated 🎤 microphone capture, and ⚡ Barge-In interruption button.
