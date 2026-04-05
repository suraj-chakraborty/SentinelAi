"""
sentinel/app/intelligence.py
────────────────────────────
High-level AI interpretation and generation with multi-model fallback.
"""

import os
import httpx
import logging
import threading
from typing import Optional, List
import json
import re

try:
    from googlesearch import search
    _GOOG_SEARCH = True
except ImportError:
    _GOOG_SEARCH = False

from sentinel.app.state import get_state

logger = logging.getLogger("SentinelIntelligence")

# ── Tools ────────────────────────────────────────────────────────────────────

def search_the_web(query: str, num_results: int = 4) -> str:
    """Fetch search results from Google."""
    if not query: return "No query provided."
    logger.info(f"Searching web for: {query}")
    
    if not _GOOG_SEARCH:
        return f"Searching for '{query}'... (googlesearch-python not installed)"
    
    try:
        results = []
        for url in search(query, num_results=num_results):
            results.append(url)
        
        if not results:
            return "No results found."
            
        return "Search Results:\n" + "\n".join([f"- {u}" for u in results])
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search failed: {e}"

# ── Gemini Generation (Core) ────────────────────────────────────────────────

def gemini_generate(prompt: str, conv=None, emotion: str = "Neutral", 
                    model: str = "gemini-2.0-flash-exp", 
                    image_b64: str = None) -> str:
    """
    Send a prompt to Gemini 2.0 Flash (primary), falling back to OpenRouter or Ollama on failure.
    Supports multimodal inputs if image_b64 is provided.
    """
    # 1. Try Gemini primary
    key = os.getenv("GEMINI_API_KEY")
    if key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        contents = []
        if conv:
            try:
                for turn in list(conv._history):
                    role = "model" if turn.role == "assistant" else "user"
                    contents.append({"role": role, "parts": [{"text": turn.content}]})
            except Exception:
                pass
        
        if not contents:
            contents.append({"role": "user", "parts": [{"text": "You are SentinelAI, an advanced personal AI assistant. Be concise."}]})
        
        user_text = f"[User emotion: {emotion}] {prompt}" if emotion and emotion != "Neutral" else prompt
        
        parts = [{"text": user_text}]
        if image_b64:
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": image_b64
                }
            })
            
        contents.append({"role": "user", "parts": parts})

        body = {"contents": contents, "generationConfig": {"maxOutputTokens": 400, "temperature": 0.7}}
        try:
            with httpx.Client(timeout=30) as client:
                r = client.post(url, json=body)
                if r.status_code == 429:
                    logger.warning("Gemini 429 Quota Exceeded. Falling back...")
                else:
                    j = r.json()
                    text = j.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                    if text: return text
        except Exception as e:
            logger.warning(f"Gemini call failed: {e}")

    # 2. Fallback to OpenRouter
    or_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_AI_API_KEY")
    if or_key:
        try:
            logger.info("Attempting OpenRouter fallback...")
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {"Authorization": f"Bearer {or_key}", "Content-Type": "application/json"}
            messages = [{"role": "user", "content": prompt}]
            if conv:
                try: messages = conv.build_messages(prompt, emotion=emotion)
                except Exception: pass
            
            payload = {"model": "google/gemini-3-flash-preview", "messages": messages, "max_tokens": 400}
            with httpx.Client(timeout=30) as client:
                r = client.post(url, json=payload, headers=headers)
                j = r.json()
                text = j.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if text: return text
        except Exception as e:
            logger.warning(f"OpenRouter fallback failed: {e}")

    return "I'm having trouble connecting to all of my intelligence engines. Please check your internet or API keys."

# ── Command Interpretation ──────────────────────────────────────────────────

def interpret_command(command: str, emotion: str = "Neutral") -> str:
    """
    Routes a command to the best available LLM with full conversation history.
    """
    if not command: return ""
    
    # 1. Search Intent detection (Simple heuristic)
    cmd = command.lower()
    search_results = ""
    if any(k in cmd for k in ("search", "find online", "who is", "what is the latest")):
        search_results = search_the_web(command)
    
    # 2. Desktop Automation Intent (Notepad)
    if "notepad" in cmd and any(k in cmd for k in ("write", "type", "open", "create")):
        from sentinel.core.computer_use_agent import ComputerUseAgent
        agent = ComputerUseAgent(speak_fn=lambda t: None) # Silent auth for agent
        # We don't block here, just trigger it
        threading.Thread(target=agent.execute, args=(command,), daemon=True).start()
        return "Of course. I'm opening Notepad to handle that for you now."

    # 3. Load conversation manager
    try:
        from sentinel.core.conversation import get_conversation
        conv = get_conversation()
    except Exception:
        conv = None

    full_prompt = f"""User Intent: {command}
Primary router failed to execute this command. 

Troubleshoot the user's request for their local PC:
1. If the user wants to open an app, find the likely executable path.
2. If they want a system action, suggest a PowerShell/CMD command.
3. If it's a general question, answer it.

If you find a working path or command, end your response with: [EXECUTE: <command_or_path>]
Otherwise, just provide a helpful response.

Search Context (if any):
{search_results}
"""
    resp = gemini_generate(full_prompt, conv=conv, emotion=emotion)
    
    if resp and conv:
        conv.add_turn("user", command, emotion=emotion)
        conv.add_turn("assistant", resp)
        
    return resp
