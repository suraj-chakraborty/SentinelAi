SentinelAI Commands and Usage Guide

This guide lists the supported voice/text commands and how to use the app in practice.

Quick Start
- Set required keys: PVPORCUPINE_PRIVATE_KEY and GEMINI_API_KEY in .env or the first-run wizard.
- Run: python sentinal_ai.py
- Say the wake word, then speak a command.
- Optional: open the dashboard at http://127.0.0.1:8000/

Help
- list commands | help commands | what can you do
  - Speaks a short capability summary.

Modes and Session Control
- start | start agent | agent start | enable agent
  - Enables agent mode (each sentence becomes a step).
- stop | stop agent | agent stop | disable agent
  - Disables agent mode.
- stop listening | pause listening | do not listen
  - Pauses listening for 5 minutes.
- start listening | resume listening
  - Resumes listening immediately.
- enable entertainment | entertain me | talk mode
  - Enables conversational replies.
- disable entertainment | quiet mode | stop talking
  - Disables conversational replies.

Agent Mode Steps
- install <app>
  - Prepares a Winget install (requires confirm).
- execute <command>
  - Prepares a shell command (requires confirm).
- confirm | yes
  - Executes the pending step.

App and Web Shortcuts
- open <app name>
  - Opens the requested application.
  - Examples: open chrome, open vs code, open word.
- open browser
  - Opens Google in the default browser.
- open gmail
  - Opens Gmail in the browser.
- open youtube
  - Opens YouTube.
- open facebook
  - Opens Facebook.
- open github
  - Opens GitHub.
- open settings
  - Opens Windows Settings.
- open task manager | task manager
  - Opens Task Manager.
- open powershell
  - Opens PowerShell.
- open vscode | open vs code
  - Opens Visual Studio Code.
- open notepad | notepad
  - Opens Notepad.
- open chrome | chrome
  - Opens a Google search for the remaining text.
- open gpt
  - Opens ChatGPT in a browser automation flow.

Secrets Vault (Encrypted)
- password for <name> is <value>
- variable <name> is <value>
  - Stores a secret under the given name.
- password for <name> give me | what is | show
- variable <name> give me | what is | show
  - Retrieves a secret and speaks it back.

Notes
- remember note <text>
- save note <text>
  - Stores a note.
- list notes | show notes
  - Lists saved notes.
- forget note <number>
  - Deletes a specific note by number.

Knowledge Base (RAG)
- remember this <text>
- save to knowledge <text>
- add to memory <text>
  - Stores information in the knowledge base.
- what do you know about <topic>
- search memory <topic>
- find info <topic>
  - Searches the knowledge base and returns matching info.

Files and Documents
- summarize my recent files
- scan documents
  - Scans recent PDFs/DOCX files and summarizes them.
- what was in the pdf <query>
- search document <query>
  - Searches indexed document content.

System and Health
- system status
- temperature
- cpu temp
  - Reads CPU/GPU temperature if available.
- system info
  - Displays detailed system information.
- ip address | show ip
  - Displays the current IPv4 address.
- battery report
  - Generates and opens a Windows battery report.

Power Actions
- shutdown
  - Shuts down the system.
- restart
  - Restarts the system.

Window Management
- minimize window | window minimize
- maximize window | window maximize
- close window | window close
- snap left | window left
- snap right | window right
- move window to left monitor | window left monitor
- move window to right monitor | window right monitor

Calendar
- upcoming meets
- google meet
- calendar
  - Lists upcoming Google Meet events.

Vision / Screen
- what is on my screen
- analyze screen
- describe window
  - Captures the screen and returns a vision summary.

Automation (GUI)
- move mouse
- click
- type text
- press key
  - Sends a GUI automation task to the system.

Web Agents
- book ticket
- search for <query>
- find <query>
  - Triggers web-agent flows for browsing or booking.
- book <task>
- order <task>
- autonomous search <task>
  - Triggers the Playwright agent flow.

IoT Control
- turn on <device>
- turn off <device>
- control light <device>
- run scene <scene>
  - Controls Home Assistant devices or runs scenes.

Daily Brief
- morning brief
- daily brief
- status report
  - Generates a daily summary.

Agentic Goals (Multi-step)
- achieve goal: <goal>
- autonomous task: <goal>
  - Breaks down and executes a multi-step task.

LLM Routing
- ask gemini <question>
- use gemini <question>
  - Routes the prompt to Gemini.

Maintenance
- clean downloads
  - Deletes files in Downloads.
- find large files
  - Lists the largest files in Downloads.

Notes
- Wake word uses Porcupine; custom keyword files can be used if bundled.
- Some features require external services or local installs: Google APIs, Playwright browsers, Ollama, Home Assistant, etc.

---

## 🌟 Sentinel AI-OS Vanguard Commands (Latest Updates)

### Omnipresent Perception Stream
- **start perception** | **turn on awareness**
  - Activates the continuous background screen-recording and context-window daemon.
- **stop perception** | **turn off awareness**
  - Deactivates the perception stream.

### Multi-Agent Swarm Intelligence
- **swarm <task>** | **delegate task <task>** | **multi-agent <task>**
  - Spawns three specialized sub-agents (Researcher, Senior Dev, Editor) to collaborate on a complex problem.

### Computer Use (Desktop Automation)
- **computer use <task>**
  - Triggers the visual desktop automation agent which physically moves the mouse and types to achieve the specified goal.

### Secure Code Sandbox
- **run this code** | **execute code** | **run python**
  - Executes dynamic python code safely through the 3-tier isolated RestrictedPython/Docker framework.

---

## 🚀 Sci-Fi Tier Commands (Hardware & Experimental)

### Embodied AI (Robotics)
- **robot <command>** | **automata <command>** | **rover <command>**
  - Transmits translated JSON kinematic directions over UDP port 4242 to any listening physical IoT rover.

### Telepathic AI (BCI Headsets)
- **connect headset** | **start bci**
  - Connects to an external Emotiv/Muse EEG headset via LSL to monitor stress (Beta wave) spikes.

### Zero-Trust Cyber Proxy
- **enable proxy shield** | **start zero trust**
  - Boots up the lightweight `asyncio` local proxy server on port 8080. Deep-inspects incoming HTML/JS payloads via LLM heuristics to block phishing attacks on the fly.

### Quantum Compute Offloading
- **quantum optimization <task>** | **calculate using qpu <task>**
  - Rewrites complex routing problems into OpenQASM 3.0 circuits via LLM and queues them directly to IBM Quantum Cloud's REST API.
