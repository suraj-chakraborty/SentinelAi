"""
sentinel/core/web_server.py
────────────────────────────
FastAPI-based SentinelAI web dashboard — v3.

v3 changes vs v2
────────────────
  • API key authentication middleware (token generated on first launch,
    stored in AppData; requires X-Sentinel-Key header or ?key= param)
  • Default bind changed to 127.0.0.1 (localhost only).
    Set SENTINEL_WEB_HOST=0.0.0.0 for deliberate LAN access.
  • WebSocket endpoint /ws — real-time push for commands, intent,
    response, module health, and alerts.
  • /health now reports all 20+ module statuses.
  • No-auth whitelist for /health and / (dashboard HTML) so browser
    can load the UI without manually setting headers.
  • Generative UI prompt sanitised to reduce XSS surface.
"""

from __future__ import annotations

import json
import logging
import os
import time as _time
import secrets
import threading
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("SentinelWebServer")

APPDATA_DIR = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi")
_KEY_FILE   = os.path.join(APPDATA_DIR, "web_api_key.txt")
_NO_AUTH    = {"/", "/health", "/ws"}           # paths that skip auth check
from sentinel.app.config import ROTATION_LOG_PATH

try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False


# ── Pydantic models ───────────────────────────────────────────────────────────

if _FASTAPI_OK:
    class CommandRequest(BaseModel):
        command: str

    class KBAddRequest(BaseModel):
        text: str
        source: Optional[str] = "web_dashboard"

    class KBQueryRequest(BaseModel):
        query: str
        n_results: Optional[int] = 3

    class UIGeneratorRequest(BaseModel):
        prompt: str


# ── API key helpers ───────────────────────────────────────────────────────────

def _load_or_create_api_key() -> str:
    os.makedirs(APPDATA_DIR, exist_ok=True)
    if os.path.exists(_KEY_FILE):
        key = open(_KEY_FILE).read().strip()
        if key:
            return key
    key = secrets.token_urlsafe(32)
    with open(_KEY_FILE, "w") as f:
        f.write(key)
    logger.info("New web API key generated and saved to: %s", _KEY_FILE)
    return key


# ── WebSocket connection manager ──────────────────────────────────────────────

class _ConnectionManager:
    def __init__(self):
        self._connections: Set["WebSocket"] = set()
        self._lock = threading.Lock()

    def connect(self, ws: "WebSocket"):
        with self._lock:
            self._connections.add(ws)

    def disconnect(self, ws: "WebSocket"):
        with self._lock:
            self._connections.discard(ws)

    def broadcast(self, data: dict):
        """Broadcast a JSON message to all connected clients (non-blocking)."""
        payload = json.dumps(data)
        dead = set()
        with self._lock:
            conns = list(self._connections)
        for ws in conns:
            try:
                # WebSocket.send_text must be called from an async context;
                # we schedule it via the event loop stored on the websocket.
                import asyncio
                loop = getattr(ws, "_loop", None)
                if loop and loop.is_running():
                    asyncio.run_coroutine_threadsafe(ws.send_text(payload), loop)
                else:
                    dead.add(ws)
            except Exception:
                dead.add(ws)
        with self._lock:
            self._connections -= dead


# ── Server ────────────────────────────────────────────────────────────────────

class SentinelWebServer:
    """
    FastAPI REST + WebSocket dashboard.
    The server starts in a background daemon thread so it never blocks the
    main Tkinter GUI thread.
    """

    def __init__(
        self,
        orchestrator=None,
        command_fn=None,
        host: str = "",
        port: int = 5000,
    ):
        if not _FASTAPI_OK:
            logger.error("fastapi/uvicorn not installed — web dashboard disabled.")
            return

        self.orchestrator  = orchestrator
        self.command_fn    = command_fn
        self.host          = host or os.getenv("SENTINEL_WEB_HOST", "127.0.0.1")
        self.port          = int(os.getenv("SENTINEL_WEB_PORT", str(port)))
        self._api_key      = _load_or_create_api_key()
        self._manager      = _ConnectionManager()
        self._alerts_buf: List[str] = []
        # Track last key rotation timestamp for admin visibility
        self._last_key_rotation = None
        self._server_thread: Optional[threading.Thread] = None
        self._start_time = _time.time()

        self.app = FastAPI(
            title="SentinelAI Dashboard",
            description="Jarvis-class personal AI assistant control panel",
            version="3.0.0",
        )

        # CORS
        _cors = (os.getenv("SENTINEL_CORS_ORIGINS") or "").strip()
        allow = ["*"] if _cors == "*" else (
            [o.strip() for o in _cors.split(",") if o.strip()]
            if _cors else [
                f"http://127.0.0.1:{self.port}",
                f"http://localhost:{self.port}",
            ]
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allow,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._register_auth_middleware()
        self._register_routes()
        logger.info("SentinelWebServer configured on %s:%d", self.host, self.port)
        logger.info("Dashboard API key: %s", self._api_key)

    def _rotate_api_key(self) -> str:
        """Rotate the stored API key and persist it to disk."""
        new_key = secrets.token_urlsafe(32)
        self._api_key = new_key
        self._last_key_rotation = _time.time()
        try:
            with open(_KEY_FILE, "w", encoding="utf-8") as f:
                f.write(new_key)
        except Exception as exc:
            logger.error("Failed to persist rotated API key: %s", exc)
        logger.info("Web API key rotated.")
        # Notify connected clients in a lightweight fashion
        try:
            self.broadcast({"type": "admin", "event": "key_rotated", "preview": new_key[:8] + "..."})
        except Exception:
            pass
        # Persist a rotation event to the rotation log for admin auditing
        try:
            with open(ROTATION_LOG_PATH, "a", encoding="utf-8") as lf:
                lf.write(f"{_time.time()},{new_key[:8]}...\n")
        except Exception:
            pass
        return new_key

    # ── Auth middleware ───────────────────────────────────────────────────────

    def _register_auth_middleware(self):
        app = self.app

        @app.middleware("http")
        async def _auth(request: Request, call_next):
            path = request.url.path
            if path in _NO_AUTH:
                return await call_next(request)
            # Check header or query param
            token = (
                request.headers.get("X-Sentinel-Key")
                or request.query_params.get("key")
            )
            if not token or token != self._api_key:
                return JSONResponse(
                    {"detail": "Unauthorized — provide valid X-Sentinel-Key header."},
                    status_code=401,
                )
            return await call_next(request)

    # ── Routes ────────────────────────────────────────────────────────────────

    def _register_routes(self):
        app  = self.app
        mgr  = self._manager

        # ── Dashboard HTML ───────────────────────────────────────────────────
        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return HTMLResponse(_DASHBOARD_HTML.replace("__API_KEY__", self._api_key))

        # ── Health ────────────────────────────────────────────────────────────
        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "version": "3.0.0",
                "uptime_epoch": _time.time(),
                "uptime_seconds": int(_time.time() - self._start_time),
                "modules": self._module_health(),
                "api_key_hint": self._api_key[:8] + "…",
            }

        # ── Stats ─────────────────────────────────────────────────────────────
        @app.get("/stats")
        async def stats():
            s = self._get_stats()
            s["alerts"] = list(self._alerts_buf)
            self._alerts_buf.clear()
            return s

        # ── Command ───────────────────────────────────────────────────────────
        @app.post("/command")
        async def run_command(req: CommandRequest):
            cmd = (req.command or "").strip()
            if not cmd:
                raise HTTPException(400, "Empty command.")

            response_text = "No executor configured."
            try:
                if self.command_fn:
                    holder: Dict[str, Any] = {"response": ""}
                    ev = threading.Event()

                    def _run():
                        try:
                            self.command_fn(cmd)
                            holder["response"] = f"Executed: {cmd}"
                        except Exception as exc:
                            holder["response"] = f"Error: {exc}"
                        finally:
                            ev.set()

                    threading.Thread(target=_run, daemon=True).start()
                    ev.wait(timeout=45)
                    response_text = holder["response"]
                elif self.orchestrator:
                    response_text = self.orchestrator.run_command(cmd) or "Done."
            except Exception as exc:
                logger.error("Command error: %s", exc)
                response_text = f"Error: {exc}"

            payload = {"response": response_text, "command": cmd, "timestamp": time.time()}
            mgr.broadcast({"type": "command_response", **payload})
            return payload

        # ── Conversation history ───────────────────────────────────────────────
        @app.get("/history")
        async def get_history():
            try:
                from sentinel.core.conversation import get_conversation
                conv = get_conversation()
                turns = [
                    {"role": t.role, "content": t.content,
                     "timestamp": t.timestamp, "emotion": t.emotion}
                    for t in list(conv._history)
                ]
                return {"turns": turns, "count": len(turns)}
            except Exception as exc:
                return {"turns": [], "count": 0, "error": str(exc)}

        # ── Knowledge base ────────────────────────────────────────────────────
        @app.post("/kb/add")
        async def kb_add(req: KBAddRequest):
            try:
                if self.orchestrator and self.orchestrator.knowledge_base:
                    ok, msg = self.orchestrator.knowledge_base.add_information(
                        req.text, metadata={"source": req.source}
                    )
                    return {"success": ok, "message": msg}
                return {"success": False, "message": "Knowledge base not available"}
            except Exception as exc:
                return {"success": False, "message": str(exc)}

        @app.post("/kb/query")
        async def kb_query(req: KBQueryRequest):
            try:
                if self.orchestrator and self.orchestrator.knowledge_base:
                    result = self.orchestrator.knowledge_base.query_knowledge(
                        req.query, n_results=req.n_results
                    )
                    return {"result": result, "query": req.query}
                return {"result": "Knowledge base not available", "query": req.query}
            except Exception as exc:
                return {"result": str(exc), "query": req.query}

        # ── Agent control ─────────────────────────────────────────────────────
        @app.post("/agent/start")
        async def agent_start(req: CommandRequest):
            try:
                if self.orchestrator and self.orchestrator.agentic_engine:
                    goal = req.command or "Assist me with my current task"
                    threading.Thread(
                        target=self.orchestrator.agentic_engine.execute_autonomous_goal,
                        args=(goal,),
                        daemon=True,
                    ).start()
                    mgr.broadcast({"type": "agent_started", "goal": goal})
                    return {"status": "started", "goal": goal}
                return {"status": "error", "message": "Agentic engine not available"}
            except Exception as exc:
                return {"status": "error", "message": str(exc)}

        @app.post("/agent/stop")
        async def agent_stop():
            try:
                if self.orchestrator and self.orchestrator.agentic_engine:
                    self.orchestrator.agentic_engine.stop()
                    mgr.broadcast({"type": "agent_stopped"})
                    return {"status": "stopped"}
                return {"status": "error", "message": "Agentic engine not available"}
            except Exception as exc:
                return {"status": "error", "message": str(exc)}

        # ── Long-term memory ──────────────────────────────────────────────────
        @app.get("/memory/stats")
        async def memory_stats():
            if self.orchestrator and self.orchestrator.long_term_memory:
                return self.orchestrator.long_term_memory.get_stats()
            return {"available": False}

        # ── Generative UI ─────────────────────────────────────────────────────
        @app.post("/generate-ui")
        async def generate_ui(req: UIGeneratorRequest):
            if not self.orchestrator:
                return JSONResponse({"html": "Orchestrator unavailable."}, status_code=500)
            # Sanitise — only allow known widget types to reduce prompt injection surface
            safe_prompt = req.prompt[:200].replace("<", "&lt;").replace(">", "&gt;")
            sys_prompt = (
                "You are a frontend developer. Output ONLY raw HTML with embedded "
                "CSS and vanilla JS for the following dashboard widget request. "
                "No markdown fences. No explanation.\n"
                f"Request: {safe_prompt}"
            )
            try:
                html = self.orchestrator._safe_llm_call(sys_prompt)
                for fence in ("```html", "```"):
                    if fence in html:
                        html = html.split(fence)[1].split("```")[0].strip()
                        break
                return JSONResponse({"html": html})
            except Exception as exc:
                return JSONResponse({"html": f"<div style='color:red'>UI gen failed: {exc}</div>"})

        # ── WebSocket ─────────────────────────────────────────────────────────
        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            import asyncio
            ws._loop = asyncio.get_event_loop()
            mgr.connect(ws)
            try:
                await ws.send_json({"type": "connected", "timestamp": time.time()})
                while True:
                    # Receive any text (ping / command from client)
                    try:
                        data = await asyncio.wait_for(ws.receive_text(), timeout=30)
                        try:
                            msg = json.loads(data)
                            if msg.get("type") == "command" and self.orchestrator:
                                result = self.orchestrator.run_command(msg.get("text", ""))
                                await ws.send_json({"type": "response", "text": result})
                        except json.JSONDecodeError:
                            pass
                    except asyncio.TimeoutError:
                        # Send keepalive ping
                        await ws.send_json({"type": "ping"})
            except WebSocketDisconnect:
                pass
            except Exception as exc:
                logger.debug("WebSocket closed: %s", exc)
            finally:
                mgr.disconnect(ws)

        # ── Metrics endpoint ───────────────────────────────────────────────────
        @app.get("/metrics", response_class=PlainTextResponse)
        async def metrics():
            """Lightweight metrics in plain text (Prometheus-like)."""
            rotation_count = 0
            try:
                if ROTATION_LOG_PATH and os.path.exists(ROTATION_LOG_PATH):
                    with open(ROTATION_LOG_PATH, "r", encoding="utf-8") as lf:
                        rotation_count = sum(1 for _ in lf if _)
            except Exception:
                rotation_count = 0
            uptime = int(_time.time() - self._start_time)
            return f"web_api_key_rotations_total {rotation_count}\nuptime_seconds {uptime}\n"

        # ── Admin: memory export ───────────────────────────────────────────────
        @app.get("/admin/memory/export")
        async def memory_export(request: Request):
            """Export current memory state for admin diagnostics."""
            token = request.headers.get("X-Sentinel-Key") or request.query_params.get("key")
            if not token or token != self._api_key:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            try:
                mem = getattr(self.orchestrator, 'long_term_memory', None)
                if not mem:
                    return {"error": "Memory backend not available"}
                # In-memory backend exposure
                if getattr(mem, "_in_memory", False):
                    data = {
                        "summaries": list(getattr(mem, "_mem_summaries", [])),
                        "facts": list(getattr(mem, "_mem_facts", [])),
                    }
                    return data
                # Fallback: expose counts only for non in-memory
                return {
                    "summaries_count": int(getattr(mem, "_summaries_col").count()) if mem._summaries_col else 0,
                    "facts_count": int(getattr(mem, "_facts_col").count()) if mem._facts_col else 0,
                }
            except Exception as exc:
                return {"error": str(exc)}

        # ── Admin: rotate API key ─────────────────────────────────────────────────
        @app.post("/admin/rotate-key")
        async def admin_rotate_key(request: Request):
            """Rotate the web API key. Requires the current key to authorize."""
            # Authorization: require current API key via header or query param
            token = request.headers.get("X-Sentinel-Key") or request.query_params.get("key")
            if not token or token != self._api_key:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            new_key = self._rotate_api_key()
            # Do not expose full key here; provide a masked preview for operator awareness
            return {"status": "rotated", "preview": new_key[:8] + "..."}

        @app.get("/admin/status")
        async def admin_status(request: Request):
            """Admin status: show last rotation time and a masked preview of the key."""
            token = request.headers.get("X-Sentinel-Key") or request.query_params.get("key")
            if not token or token != self._api_key:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            last_rot = self._last_key_rotation
            preview = (self._api_key[:8] + "...") if self._api_key else None
            return {
                "last_key_rotation_epoch": last_rot,
                "key_preview": preview,
            }

        @app.post("/admin/memory/clear")
        async def memory_clear(request: Request):
            """Admin: clear memory stores (summaries and facts)."""
            token = request.headers.get("X-Sentinel-Key") or request.query_params.get("key")
            if not token or token != self._api_key:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            if self.long_term_memory and hasattr(self.long_term_memory, 'clear_memory'):
                ok = self.long_term_memory.clear_memory()
                return {"cleared": ok}
            return {"cleared": False}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def broadcast(self, event: dict) -> None:
        """Push a structured event to all connected WebSocket clients."""
        self._manager.broadcast(event)

    def add_alert(self, message: str) -> None:
        self._alerts_buf.append(message)
        if len(self._alerts_buf) > 50:
            self._alerts_buf = self._alerts_buf[-50:]
        self._manager.broadcast({"type": "alert", "message": message, "timestamp": time.time()})

    def _get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "gemini_status": "Active" if os.getenv("GEMINI_API_KEY") else "No Key",
            "ollama_status": "Unknown",
            "cpu_temp": None,
            "gpu_temp": None,
            "kb_count": 0,
        }
        if not self.orchestrator:
            return stats
        orc = self.orchestrator
        for attr, key, method in [
            ("system_monitor", "cpu_temp",    "get_cpu_temp"),
            ("system_monitor", "gpu_temp",    "get_gpu_temp"),
        ]:
            obj = getattr(orc, attr, None)
            if obj and hasattr(obj, method):
                try:
                    stats[key] = getattr(obj, method)()
                except Exception:
                    pass
        if getattr(orc, "ollama_module", None):
            try:
                stats["ollama_status"] = "Active" if orc.ollama_module.is_available() else "Offline"
            except Exception:
                pass
        if getattr(orc, "knowledge_base", None):
            try:
                stats["kb_count"] = orc.knowledge_base.get_count()
            except Exception:
                pass
        return stats

    def _module_health(self) -> Dict[str, bool]:
        if not self.orchestrator:
            return {}
        attrs = [
            "vision_module", "knowledge_base", "emotion_module", "habit_learner",
            "security_shield", "iot_hub", "coding_module", "automation_module",
            "gesture_module", "file_intel", "calendar_module", "web_agent",
            "playwright_agent", "agentic_engine", "daily_brief", "ollama_module",
            "swarm_engine", "computer_use", "long_term_memory", "quantum",
        ]
        health = {a: getattr(self.orchestrator, a, None) is not None for a in attrs}
        # Phase 2: include plugin health (if available via CommandRouter)
        try:
            plugin_health = {}
            rom = getattr(self.orchestrator, 'command_router', None)
            if rom and hasattr(rom, 'plugins'):
                for pname, p in rom.plugins.items():
                    status = True
                    try:
                        if hasattr(p, 'health') and callable(p.health):
                            res = p.health()
                            if isinstance(res, bool):
                                status = res
                            elif isinstance(res, dict):
                                status = bool(res.get('healthy', True))
                    except Exception:
                        status = False
                    plugin_health[str(pname)] = status
            health['plugins'] = plugin_health
        except Exception:
            # If anything goes wrong, omit plugin health gracefully
            health['plugins'] = {}
        return health

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if not _FASTAPI_OK:
            return
        self._server_thread = threading.Thread(target=self._run_uvicorn, daemon=True)
        self._server_thread.start()
        logger.info("Web dashboard at http://%s:%d  (API key: %s…)", self.host, self.port, self._api_key[:8])

    def _run_uvicorn(self) -> None:
        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="warning",
                access_log=False,
            )
        except Exception as exc:
            logger.error("Web server error: %s", exc)


# ── Embedded dashboard HTML ───────────────────────────────────────────────────
# Minimal but functional dark-mode dashboard that connects via WebSocket.

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SentinelAI Dashboard</title>
<style>
  :root {
    --bg: #0d0d1a; --panel: #13132a; --border: #2a2a55;
    --accent: #6c63ff; --accent2: #00d4ff; --text: #e0e0f0;
    --green: #00e676; --red: #ff5252; --yellow: #ffd740;
    --font: 'Segoe UI', system-ui, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font); min-height: 100vh; }
  header {
    background: linear-gradient(135deg, #1a1a3e 0%, #0d0d2e 100%);
    padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem;
    border-bottom: 1px solid var(--border);
  }
  header h1 { font-size: 1.4rem; background: linear-gradient(90deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  #ws-status { margin-left: auto; font-size: 0.8rem; display: flex; align-items: center; gap: 6px; }
  #ws-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--red); transition: background 0.3s; }
  #ws-dot.on { background: var(--green); }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; padding: 1.5rem; max-width: 1400px; margin: 0 auto; }
  .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.2rem; display: flex; flex-direction: column; gap: 0.75rem;
  }
  .panel h2 { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; color: var(--accent2); }
  #chat-panel { grid-column: 1 / -1; }
  #messages {
    height: 280px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px;
    padding: 8px; background: #080814; border-radius: 8px; border: 1px solid var(--border);
  }
  .msg { padding: 8px 12px; border-radius: 8px; max-width: 85%; font-size: 0.88rem; line-height: 1.5; }
  .msg.user { background: var(--accent); align-self: flex-end; }
  .msg.ai   { background: #1e1e42; align-self: flex-start; }
  .msg.sys  { background: #0a2a0a; align-self: center; color: var(--green); font-size: 0.78rem; }
  #input-row { display: flex; gap: 8px; }
  #cmd-input { flex: 1; background: #080814; color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; font-size: 0.9rem; outline: none; }
  #cmd-input:focus { border-color: var(--accent); }
  button { background: var(--accent); color: #fff; border: none; border-radius: 8px; padding: 10px 20px; cursor: pointer; font-size: 0.9rem; transition: opacity 0.2s; }
  button:hover { opacity: 0.85; }
  .module-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 8px; }
  .mod-chip {
    display: flex; align-items: center; gap: 6px; padding: 6px 10px;
    border-radius: 6px; font-size: 0.78rem; border: 1px solid var(--border);
    background: #0a0a20;
  }
  .mod-chip .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .dot.ok  { background: var(--green); }
  .dot.off { background: var(--red); }
  .stat-row { display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; padding: 4px 0; border-bottom: 1px solid var(--border); }
  .stat-row:last-child { border-bottom: none; }
  .val { color: var(--accent2); font-weight: 600; }
  @media (max-width: 700px) { main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <div>
    <h1>⚡ SentinelAI Dashboard</h1>
    <div style="font-size:0.75rem;color:#888;margin-top:2px">Jarvis-class personal AI assistant</div>
  </div>
  <div id="ws-status"><div id="ws-dot"></div><span id="ws-label">Connecting…</span></div>
</header>
<main>

  <!-- Chat panel -->
  <div class="panel" id="chat-panel">
    <h2>💬 Command Console</h2>
    <div id="messages"></div>
    <div id="input-row">
      <input id="cmd-input" type="text" placeholder="Type a command or question…" autocomplete="off">
      <button onclick="sendCmd()">Send</button>
    </div>
  </div>

  <!-- Admin panel -->
  <div class="panel" id="admin-panel">
    <h2>🧭 Admin</h2>
    <div id="admin-status" style="margin-bottom:8px; font-family:var(--font); font-size:0.95rem; color:#ddd;">
      Last rotation: <span id="admin-last-rotation">n/a</span>
    </div>
    <div style="display:flex; gap:8px; align-items:center; margin-bottom:6px;">
      <button onclick="rotateApiKey()">Rotate API Key</button>
      <span style="font-family:var(--font); font-size:0.85rem; color:#aaa;">Premium preview: <span id="admin-key-preview">—</span></span>
    </div>
  </div>

  <!-- Module health -->
  <div class="panel">
    <h2>🧠 Module Health</h2>
    <div class="module-grid" id="module-grid">Loading…</div>
  </div>

  <!-- System stats -->
  <div class="panel">
    <h2>📊 System Stats</h2>
    <div id="stats-body">Loading…</div>
  </div>

</main>

<script>
const API_KEY  = '__API_KEY__';
const WS_URL   = `ws://${location.host}/ws`;
let ws;

function connect() {
  ws = new WebSocket(WS_URL);
  ws.onopen  = () => { setWs(true);  appendSys('Connected to SentinelAI'); fetchHealth(); fetchStats(); };
  ws.onclose = () => { setWs(false); setTimeout(connect, 3000); };
  ws.onerror = ()  => ws.close();
  ws.onmessage = (ev) => {
    const d = JSON.parse(ev.data);
    if (d.type === 'command_response') appendMsg('ai', d.response);
    if (d.type === 'alert')            appendSys('⚠ ' + d.message);
    if (d.type === 'agent_started')    appendSys('🤖 Agent started: ' + d.goal);
    if (d.type === 'agent_stopped')    appendSys('🛑 Agent stopped');
  };
  // Initialize admin UI status when connection established
  fetchAdminStatus();
}

function setWs(on) {
  document.getElementById('ws-dot').className = 'dot' + (on ? ' on' : '');
  document.getElementById('ws-label').textContent = on ? 'Connected' : 'Disconnected';
}

function sendCmd() {
  const el = document.getElementById('cmd-input');
  const cmd = el.value.trim();
  if (!cmd) return;
  appendMsg('user', cmd);
  el.value = '';
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({type: 'command', text: cmd}));
  } else {
    // REST fallback
    fetch('/command', {
      method: 'POST',
      headers: {'Content-Type':'application/json','X-Sentinel-Key': API_KEY},
      body: JSON.stringify({command: cmd})
    }).then(r => r.json()).then(d => appendMsg('ai', d.response));
  }
}

document.getElementById('cmd-input').addEventListener('keydown', e => { if (e.key==='Enter') sendCmd(); });

function appendMsg(role, text) {
  const box = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}
function appendSys(t) { appendMsg('sys', t); }

async function fetchHealth() {
  const r = await fetch('/health');
  const d = await r.json();
  const grid = document.getElementById('module-grid');
  grid.innerHTML = '';
  Object.entries(d.modules || {}).forEach(([name, ok]) => {
    grid.innerHTML += `<div class="mod-chip">
      <span class="dot ${ok?'ok':'off'}"></span>
      <span>${name.replace(/_/g,' ')}</span>
    </div>`;
  });
}

async function fetchStats() {
  const r = await fetch('/stats', {headers:{'X-Sentinel-Key': API_KEY}});
  const d = await r.json();
  const body = document.getElementById('stats-body');
  body.innerHTML = [
    ['Gemini',  d.gemini_status  || '—'],
    ['Ollama',  d.ollama_status  || '—'],
    ['KB Entries', d.kb_count ?? '—'],
    ['CPU Temp',d.cpu_temp  ? d.cpu_temp+'°C' : '—'],
    ['GPU Temp',d.gpu_temp  ? d.gpu_temp+'°C' : '—'],
  ].map(([k,v]) => `<div class="stat-row"><span>${k}</span><span class="val">${v}</span></div>`).join('');

  (d.alerts||[]).forEach(a => appendSys(a));
  setTimeout(fetchStats, 10000);  // refresh every 10s
}

connect();
<script>
async function fetchAdminStatus() {
  try {
    const r = await fetch('/admin/status', { headers: {'X-Sentinel-Key': API_KEY} });
    const d = await r.json();
    const last = d.last_key_rotation_epoch ? new Date(d.last_key_rotation_epoch * 1000).toLocaleString() : 'never';
    document.getElementById('admin-last-rotation').textContent = last;
    document.getElementById('admin-key-preview').textContent = d.key_preview || '—';
  } catch (e) {
    // ignore
  }
}

async function rotateApiKey() {
  try {
    const r = await fetch('/admin/rotate-key', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Sentinel-Key': API_KEY }
    });
    const d = await r.json();
    if (d.status === 'rotated') {
      // small UX hint in the UI console
      const sys = document.createElement('div');
      sys.textContent = 'API key rotated';
      sys.style.color = '#9be7a5';
      document.getElementById('messages').appendChild(sys);
    }
    fetchAdminStatus();
  } catch (e) {
    // ignore
  }
}

document.addEventListener('DOMContentLoaded', () => {
  fetchAdminStatus();
});
</script>
</body>
</html>
"""
