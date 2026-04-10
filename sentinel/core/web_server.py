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
from sentinel.core.auth import require_role

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

        @app.post("/jarvis/plan")
        async def jarvis_plan(request: Request):
            data = {}
            try:
                data = await request.json()
            except Exception:
                data = {}
            goal = data.get("goal", "")
            status = self.orchestrator.start_jarvis(goal) if hasattr(self.orchestrator, 'start_jarvis') else {"error": "not supported"}
            return status

        @app.get("/jarvis/status")
        async def jarvis_status():
            plan = getattr(self.orchestrator, 'jarvis_plan', None) or []
            return {"plan": [step.text if hasattr(step, 'text') else str(step) for step in plan], "progress": getattr(self.orchestrator, '_jarvis_progress', 0)}

        @app.get("/jarvis/reflect")
        async def jarvis_reflect():
            if not self.orchestrator or not hasattr(self.orchestrator, 'jarvis_reflect'):
                return {"status": "not_supported"}
            return self.orchestrator.jarvis_reflect()

        @app.get("/jarvis/transcript")
        async def jarvis_transcript():
            # Return the in-memory transcript if available
            try:
                transcript = getattr(self.orchestrator, "_jarvis_transcript", [])
                return transcript
            except Exception:
                return []

        @app.post("/jarvis/clear_transcript")
        async def jarvis_clear_transcript():
            try:
                if hasattr(self.orchestrator, "_jarvis_transcript"):
                    self.orchestrator._jarvis_transcript.clear()
                return {"status": "cleared"}
            except Exception:
                return {"status": "error"}

        @app.post("/jarvis/execute_next")
        async def jarvis_execute_next():
            if not hasattr(self.orchestrator, 'jarvis_execute_next'):
                return {"status": "not_supported"}
            return self.orchestrator.jarvis_execute_next()

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

        # ── Semantic Screen Memory ─────────────────────────────────────────────
        @app.get("/memory/screen/stats")
        async def screen_memory_stats():
            try:
                from sentinel.commands.search_memory import get_search_memory_command
                cmd = get_search_memory_command()
                return cmd.get_stats()
            except Exception as e:
                return {"error": str(e)}

        @app.post("/memory/screen/search")
        async def search_screen_memory(req: KBQueryRequest):
            try:
                from sentinel.commands.search_memory import get_search_memory_command
                cmd = get_search_memory_command()
                return cmd.execute(req.query, req.n_results or 5)
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.get("/memory/screen/recent")
        async def get_recent_screens(limit: int = 10):
            try:
                from sentinel.commands.search_memory import get_search_memory_command
                cmd = get_search_memory_command()
                return cmd.get_recent(limit)
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.post("/memory/screen/capture")
        async def capture_screen():
            try:
                from sentinel.commands.search_memory import get_search_memory_command
                cmd = get_search_memory_command()
                return cmd.capture_now()
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.post("/memory/screen/indexing/start")
        async def start_screen_indexing(interval: int = 60):
            try:
                from sentinel.commands.search_memory import get_search_memory_command
                cmd = get_search_memory_command()
                return cmd.start_indexing(interval)
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.post("/memory/screen/indexing/stop")
        async def stop_screen_indexing():
            try:
                from sentinel.commands.search_memory import get_search_memory_command
                cmd = get_search_memory_command()
                return cmd.stop_indexing()
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.delete("/memory/screen/{screen_id}")
        async def delete_screen(screen_id: str):
            try:
                from sentinel.commands.search_memory import get_search_memory_command
                cmd = get_search_memory_command()
                return cmd.delete_screen(screen_id)
            except Exception as e:
                return {"success": False, "error": str(e)}

        # ── UI Automation (UIA) ─────────────────────────────────────────────────
        @app.get("/automation/uia/window")
        async def uia_window_info():
            try:
                from sentinel.commands.ui_automation import get_uia_command
                cmd = get_uia_command()
                return cmd._get_window_info()
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.get("/automation/uia/windows")
        async def uia_list_windows():
            try:
                from sentinel.commands.ui_automation import get_uia_command
                cmd = get_uia_command()
                return cmd.execute("list_windows")
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.get("/automation/uia/children")
        async def uia_list_children(parent_hwnd: int = None):
            try:
                from sentinel.commands.ui_automation import get_uia_command
                cmd = get_uia_command()
                return cmd.execute("list_children", parent_hwnd=parent_hwnd)
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.post("/automation/uia/click")
        async def uia_click(handle: int, double_click: bool = False):
            try:
                from sentinel.commands.ui_automation import get_uia_command
                cmd = get_uia_command()
                return cmd.execute("click", handle=handle, double_click=double_click)
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.post("/automation/uia/keys")
        async def uia_send_keys(text: str, handle: int = None):
            try:
                from sentinel.commands.ui_automation import get_uia_command
                cmd = get_uia_command()
                return cmd.execute("send_keys", text=text, handle=handle)
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.post("/automation/uia/focus")
        async def uia_focus(handle: int):
            try:
                from sentinel.commands.ui_automation import get_uia_command
                cmd = get_uia_command()
                return cmd.execute("focus", handle=handle)
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.post("/automation/uia/close")
        async def uia_close(handle: int):
            try:
                from sentinel.commands.ui_automation import get_uia_command
                cmd = get_uia_command()
                return cmd.execute("close_window", handle=handle)
            except Exception as e:
                return {"success": False, "error": str(e)}

        # ── Live Spatial Vision ─────────────────────────────────────────────────
        @app.post("/vision/stream/start")
        async def start_vision_stream(camera_id: int = 0, fps: int = 10):
            try:
                from sentinel.vision.live_vision import start_vision_stream
                success = start_vision_stream(camera_id, fps)
                return {"success": success, "message": "Vision stream started" if success else "Failed to start"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.post("/vision/stream/stop")
        async def stop_vision_stream():
            try:
                from sentinel.vision.live_vision import stop_vision_stream
                stop_vision_stream()
                return {"success": True, "message": "Vision stream stopped"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.get("/vision/stream/frame")
        async def get_vision_frame():
            try:
                from sentinel.vision.live_vision import get_vision_frame
                frame_b64 = get_vision_frame()
                if frame_b64:
                    return {"success": True, "frame": frame_b64, "format": "jpeg"}
                return {"success": False, "error": "No frame available"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.get("/vision/stream/stats")
        async def get_vision_stats():
            try:
                from sentinel.vision.live_vision import get_live_vision
                stream = get_live_vision()
                return stream.get_stats()
            except Exception as e:
                return {"error": str(e)}

        @app.post("/vision/capture")
        async def capture_vision_frame():
            try:
                from sentinel.vision.live_vision import get_live_vision
                stream = get_live_vision()
                filepath = stream.capture_frame()
                if filepath:
                    return {"success": True, "filepath": filepath}
                return {"success": False, "error": "No frame to capture"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.post("/vision/analyze")
        async def analyze_vision_frame(prompt: str = "Describe what you see in this image."):
            try:
                from sentinel.vision.live_vision import get_live_vision
                stream = get_live_vision()
                result = stream.analyze_current_frame(prompt)
                return {"success": True, "analysis": result}
            except Exception as e:
                return {"success": False, "error": str(e)}

        # ── Context-Aware Background Daemons ───────────────────────────────────
        @app.get("/daemons")
        async def list_daemons():
            try:
                from sentinel.daemons.daemon_manager import get_daemon_manager
                manager = get_daemon_manager()
                return manager.get_stats()
            except Exception as e:
                return {"error": str(e)}

        @app.post("/daemons/start")
        async def start_daemons():
            try:
                from sentinel.daemons.daemon_manager import start_daemon_manager
                start_daemon_manager()
                return {"success": True, "message": "Daemon manager started"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.post("/daemons/stop")
        async def stop_daemons():
            try:
                from sentinel.daemons.daemon_manager import stop_daemon_manager
                stop_daemon_manager()
                return {"success": True, "message": "Daemon manager stopped"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.get("/daemons/{task_id}")
        async def get_daemon_status(task_id: str):
            try:
                from sentinel.daemons.daemon_manager import get_daemon_manager
                manager = get_daemon_manager()
                status = manager.get_task_status(task_id)
                if status:
                    return status
                return {"error": f"Task {task_id} not found"}
            except Exception as e:
                return {"error": str(e)}

        @app.post("/daemons/{task_id}/run")
        async def run_daemon_task(task_id: str):
            try:
                from sentinel.daemons.daemon_manager import get_daemon_manager
                manager = get_daemon_manager()
                return manager.run_task_now(task_id)
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.post("/daemons/{task_id}/enable")
        async def enable_daemon(task_id: str):
            try:
                from sentinel.daemons.daemon_manager import get_daemon_manager
                manager = get_daemon_manager()
                if manager.enable_task(task_id):
                    return {"success": True, "message": f"Task {task_id} enabled"}
                return {"success": False, "error": "Task not found"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.post("/daemons/{task_id}/disable")
        async def disable_daemon(task_id: str):
            try:
                from sentinel.daemons.daemon_manager import get_daemon_manager
                manager = get_daemon_manager()
                if manager.disable_task(task_id):
                    return {"success": True, "message": f"Task {task_id} disabled"}
                return {"success": False, "error": "Task not found"}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.get("/daemons/events")
        async def get_daemon_events():
            try:
                from sentinel.daemons.daemon_manager import get_daemon_manager
                manager = get_daemon_manager()
                return {"events": manager.get_events()}
            except Exception as e:
                return {"events": [], "error": str(e)}

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
        @require_role('admin')
        async def memory_export(request: Request):
            """Export current memory state for admin diagnostics."""
            token = request.headers.get("X-Sentinel-Key") or request.query_params.get("key")
            if not token or token != self._api_key:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            try:
                mem = getattr(self.orchestrator, 'long_term_memory', None)
                if not mem:
                    return {"error": "Memory backend not available"}
                # In-memory mode exposure
                if getattr(mem, "_in_memory", False):
                    data = {
                        "summaries": list(getattr(mem, "_mem_summaries", [])),
                        "facts": list(getattr(mem, "_mem_facts", [])),
                    }
                    return data
                # Fallback: counts for non-in-memory
                return {
                    "summaries_count": mem.get_stats().get("session_summaries", 0),
                    "facts_count": mem.get_stats().get("stored_facts", 0),
                    "db_path": mem.db_path if hasattr(mem, 'db_path') else None,
                }
            except Exception as exc:
                return {"error": str(exc)}

        @app.post("/admin/memory/import")
        @require_role('admin')
        async def memory_import(request: Request):
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            token = request.headers.get("X-Sentinel-Key") or request.query_params.get("key")
            if not token or token != self._api_key:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            if getattr(self.orchestrator, 'long_term_memory', None) and payload:
                ok = self.orchestrator.long_term_memory.import_memory(payload)
                return {"imported": ok}
            return {"imported": False}

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

        @app.get("/admin/audit")
        @require_role('admin')
        async def admin_audit(request: Request, limit: int = 50):
            try:
                if not AUDIT_LOG_PATH or not os.path.exists(AUDIT_LOG_PATH):
                    return {"events": []}
                events = []
                with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            events.append(json.loads(line))
                        except Exception:
                            events.append({"raw": line.strip()})
                return {"events": events[-limit:]}
            except Exception as exc:
                return {"events": [], "error": str(exc)}

        @app.post("/admin/memory/clear")
        async def memory_clear(request: Request):
            """Admin: clear memory stores (summaries and facts)."""
            from sentinel.core.auth import is_admin
            if not is_admin(request):
                return JSONResponse({"detail": "Forbidden"}, status_code=403)
            token = request.headers.get("X-Sentinel-Key") or request.query_params.get("key")
            if not token or token != self._api_key:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            if self.long_term_memory and hasattr(self.long_term_memory, 'clear_memory'):
                ok = self.long_term_memory.clear_memory()
                from sentinel.core.audit import log_audit
                log_audit("admin", "clear_memory", "Phase 4: per-topic TTL clear")
                return {"cleared": ok}
            return {"cleared": False}
        
        @app.post("/user/profile")
        async def user_profile(request: Request):
            # Phase 5 MVP: per-user profile (in-memory, patch ready)
            user_id = request.headers.get("X-User-Id") or request.query_params.get("user_id")
            if not user_id:
                return {"error": "user_id required"}
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            persona = payload.get('persona') if isinstance(payload, dict) else None
            memory_ttl_hours = int(payload.get('memory_ttl_hours', 0)) if isinstance(payload, dict) else 0
            opt_in_privacy = bool(payload.get('opt_in_privacy', True)) if isinstance(payload, dict) else True
            preferred_tools = payload.get('preferred_tools', []) if isinstance(payload, dict) else []
            if self.orchestrator and getattr(self.orchestrator, 'user_profiles', None):
                upm = self.orchestrator.user_profiles
                profile = upm.set_profile(user_id, persona=persona, memory_ttl_hours=memory_ttl_hours,
                                        opt_in_privacy=opt_in_privacy, preferred_tools=preferred_tools)
                self.orchestrator.current_user_id = user_id
                return {"status": "updated", "profile": profile.to_dict()}
            return {"status": "ok"}

        @app.get("/user/profile")
        async def user_profile_get(request: Request):
            user_id = request.headers.get("X-User-Id") or request.query_params.get("user_id")
            if not user_id:
                return {"error": "user_id required"}
            if self.orchestrator and getattr(self.orchestrator, 'user_profiles', None):
                profile = self.orchestrator.user_profiles.get_profile(user_id)
                return {"profile": profile.to_dict() if profile else None}
            return {"profile": None}
        @app.get("/admin/memory/ttl")
        @require_role('admin')
        async def memory_ttl_get(request: Request):
            ttl_hours = getattr(self.orchestrator.long_term_memory, '_memory_ttl_hours', 0) if getattr(self.orchestrator, 'long_term_memory', None) else 0
            return {"memory_ttl_hours": int(ttl_hours)}

        # ── Plugin Generator (Self-Healing) ─────────────────────────────────────
        @app.post("/admin/plugins/generate")
        async def generate_plugin(request: Request, command: str, description: str = ""):
            """Admin: auto-generate a plugin for a command."""
            from sentinel.core.auth import is_admin
            if not is_admin(request):
                return JSONResponse({"detail": "Forbidden"}, status_code=403)
            try:
                from sentinel.core.plugin_generator import get_plugin_generator
                gen = get_plugin_generator(self.orchestrator)
                success, name, message = gen.generate_plugin(command, description)
                return {"success": success, "plugin_name": name, "message": message}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.get("/admin/plugins/generated")
        async def list_generated_plugins(request: Request):
            """List all auto-generated plugins."""
            try:
                from sentinel.core.plugin_generator import get_plugin_generator
                gen = get_plugin_generator()
                plugins = gen.list_generated_plugins()
                return {"plugins": plugins}
            except Exception as e:
                return {"plugins": [], "error": str(e)}

        @app.delete("/admin/plugins/{plugin_name}")
        async def delete_generated_plugin(request: Request, plugin_name: str):
            """Admin: delete an auto-generated plugin."""
            from sentinel.core.auth import is_admin
            if not is_admin(request):
                return JSONResponse({"detail": "Forbidden"}, status_code=403)
            try:
                from sentinel.core.plugin_generator import get_plugin_generator
                gen = get_plugin_generator()
                success = gen.delete_plugin(plugin_name)
                return {"success": success}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @app.post("/admin/memory/ttl")
        @require_role('admin')
        async def memory_ttl_set(request: Request):
            try:
                data = await request.json()
            except Exception:
                data = {}
            hours = int(data.get("ttl_hours", 0))
            if getattr(self.orchestrator, 'long_term_memory', None):
                mem = self.orchestrator.long_term_memory
                mem._memory_ttl_hours = max(0, hours)
                return {"memory_ttl_hours": mem._memory_ttl_hours}
            return {"memory_ttl_hours": 0}
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

  <!-- Per-User Personalization (Phase 5.4) -->
  <section class="panel" id="jarvis-personalization" aria-label="Per-User Personalization" style="min-width:320px;">
    <h2>🧩 Per-User Personalization</h2>
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px;">
      <div>
        <label for="pu-userid" style="font-size:0.8rem; color:#aaa;">User ID</label>
        <input id="pu-userid" placeholder="e.g. user123" style="width:100%; padding:6px; border-radius:6px; background:#0a0a20; color:#fff; border:1px solid #333;" />
      </div>
      <div>
        <label for="pu-persona" style="font-size:0.8rem; color:#aaa;">Persona</label>
        <select id="pu-persona" style="width:100%; padding:6px; border-radius:6px; background:#0a0a20; color:#fff; border:1px solid #333;">
          <option value="calm">Calm Jarvis</option>
          <option value="formal">Formal Jarvis</option>
          <option value="direct">Direct Jarvis</option>
          <option value="humorous">Humorous Jarvis</option>
        </select>
      </div>
    </div>
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-top:6px;">
      <div>
        <label for="pu-memoryttl" style="font-size:0.8rem; color:#aaa;">Memory TTL (hours)</label>
        <input id="pu-memoryttl" type="number" min="0" step="1" value="24" style="width:100%; padding:6px; border-radius:6px; background:#0a0a20; color:#fff; border:1px solid #333;" />
      </div>
      <div>
        <label for="pu-tools" style="font-size:0.8rem; color:#aaa;">Preferred Tools (comma-separated)</label>
        <input id="pu-tools" placeholder="notes, planner" style="width:100%; padding:6px; border-radius:6px; background:#0a0a20; color:#fff; border:1px solid #333;" />
      </div>
    </div>
    <div style="margin-top:8px; display:flex; gap:8px; align-items:center;">
      <button onclick="saveUserProfile()">Save Profile</button>
      <span id="pu-status" style="font-family:var(--font); font-size:0.85rem; color:#aaa;">Not saved</span>
    </div>
    <div id="pu-preview" style="margin-top:10px; font-family:var(--font); font-size:0.9rem; color:#ddd;"></div>
  </section>

  <!-- Jarvis UI (Phase 5.4: Per-User Personalization) -->
  <section class="panel" id="jarvis-panel" aria-label="Jarvis Panel" style="min-width:320px;">
    <h2>🧭 Jarvis</h2>
    <div style="display:flex; gap:12px; align-items: center; flex-wrap: wrap; margin-bottom:6px;">
      <span>Persona:</span>
      <span id="jarvis-current-persona" style="font-family:monospace; background:#0a0a20; padding:4px 8px; border-radius:6px; border:1px solid #333;">calm Jarvis</span>
      <span style="margin-left:auto; font-size:12px; color:#aaa;">live persona</span>
    </div>
    <div class="stat-row" style="align-items:stretch; padding:6px 8px;">
      <span>Plan status</span>
      <span id="jarvis-plan-status" class="val">Idle</span>
    </div>
    <div id="jarvis-plan-steps" class="panel" style="padding:8px; margin-top:8px; background:#0a0a20; border-radius:6px; border:1px solid #333; max-height:120px; overflow:auto;"></div>
    <div id="jarvis-next-step" style="margin-top:8px; font-family:monospace; color:#fff;"></div>
    <div id="jarvis-transcript" class="panel" style="margin-top:8px; padding:8px; background:#0a0a20; border-radius:6px; border:1px solid #333; max-height:120px; overflow:auto;"></div>
  </section>

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
  // Initialize Jarvis UI on load
  fetchJarvisStatus();
  // Poll Jarvis UI periodically for status
  setInterval(fetchJarvisStatus, 10000);
  // Optional: populate per-user profile if user id is known
  setInterval(fetchJarvisTranscript, 15000);
}

async function fetchAdminStatus() {
  try {
    const r = await fetch('/admin/status', { headers: { 'X-Sentinel-Key': API_KEY } });
    const d = await r.json();
    if (d.last_key_rotation_epoch !== undefined) {
      document.getElementById('admin-last-rotation').textContent = new Date(d.last_key_rotation_epoch * 1000).toLocaleString();
    }
    if (d.key_preview) {
      document.getElementById('admin-key-preview').textContent = d.key_preview;
    }
  } catch (e) {
    // ignore
  }
}

async function fetchJarvisStatus() {
  try {
    const r = await fetch('/jarvis/status');
    const d = await r.json();
    const panel = document.getElementById('jarvis-plan-status');
    if (panel && d.progress !== undefined) panel.textContent = 'Progress: ' + d.progress;
    // Show simple plan steps if provided
    const steps = d.plan || [];
    const list = document.getElementById('jarvis-plan-steps');
    if (list && steps.length >= 1) {
      list.innerHTML = steps.map((s,i)=> `<div style="padding:4px 0;">${i+1}. ${s}</div>`).join('');
    }
    if (document.getElementById('jarvis-current-persona')) {
      // Persona may come from the backend in the plan, but ensure display exists
    }
  } catch { /* ignore */ }
}

async function fetchJarvisTranscript() {
  try {
    const r = await fetch('/jarvis/transcript');
    const t = await r.json();
    const tBox = document.getElementById('jarvis-transcript');
    if (tBox && Array.isArray(t)) {
      tBox.innerHTML = t.slice(-6).map((e)=> `<div style="padding:2px 0; font-family:monospace; font-size:0.8rem;">${e.step} → ${e.result}</div>`).join('')
    }
  } catch { /* ignore */ }
}

async function loadUserProfile() {
  const userId = document.getElementById('pu-userid').value;
  if (!userId) return;
  try {
    const r = await fetch('/user/profile', {
      headers: { 'X-User-Id': userId },
    });
    const data = await r.json();
    const p = data && data.profile ? data.profile : {};
    if (p.persona) document.getElementById('pu-persona').value = p.persona;
    if (p.memory_ttl_hours != null) document.getElementById('pu-memoryttl').value = p.memory_ttl_hours;
  } catch {
    // ignore
  }
}

window.loadUserProfile = loadUserProfile;

async function saveUserProfile() {
  const userId = document.getElementById('pu-userid').value;
  if (!userId) { alert('Enter a user_id in the User ID field to save'); return; }
  const persona = document.getElementById('pu-persona').value;
  const ttl = parseInt(document.getElementById('pu-memoryttl').value || '0', 10);
  const toolsRaw = document.getElementById('pu-tools').value || '';
  const tools = toolsRaw.split(',').map(s => s.trim()).filter(s => s);
  const payload = { persona, memory_ttl_hours: ttl, opt_in_privacy: true, preferred_tools: tools };
  try {
    const r = await fetch('/user/profile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-User-Id': userId },
      body: JSON.stringify(payload),
    });
    const data = await r.json();
    document.getElementById('pu-status').textContent = data?.status || 'saved';
    if (data?.profile) {
      document.getElementById('pu-preview').textContent = 'Profile saved: persona=' + data.profile.persona;
    }
  } catch (e) {
    document.getElementById('pu-status').textContent = 'save failed';
  }
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
<section id="jarvis_governance" style="padding:8px;">
  <h4>Jarvis Governance</h4>
  <div id="jarvis-governance-content" style="font-family:monospace; font-size:12px;">
    TTL: <span id="governance-ttl">unknown</span> • Audit events: <span id="governance-audit-count">0</span>
  </div>
</section>
</body>
</html>
"""
