"""
sentinel/core/web_server.py
────────────────────────────
FastAPI-based SentinelAI web dashboard — Tier 2 Upgrade.

Upgraded from a minimal 3-endpoint server to a full REST API with:
  - Real-time system stats (CPU, GPU, KB count, module health)
  - Full conversation history endpoint
  - Knowledge base search + add endpoints
  - Security alerts feed
  - Agent control (start/stop)
  - CORS support for external tools
"""

import os
import logging
import threading
import time
from typing import Any, Dict, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.templating import Jinja2Templates
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

logger = logging.getLogger("SentinelWebServer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "..", "templates")


# ─── Request/Response models ─────────────────────────────────────────────────


class CommandRequest(BaseModel):
    command: str


class KBAddRequest(BaseModel):
    text: str
    source: Optional[str] = "web_dashboard"


class KBQueryRequest(BaseModel):
    query: str
    n_results: Optional[int] = 3


# ─── Server class ─────────────────────────────────────────────────────────────


class SentinelWebServer:
    """
    FastAPI web dashboard for remote monitoring and command execution.
    Runs in a background daemon thread.
    """

    def __init__(self, orchestrator=None, command_fn=None, host: str = "0.0.0.0", port: int = 5000):
        if not _FASTAPI_AVAILABLE:
            logger.error("FastAPI not installed. Run: pip install fastapi uvicorn")
            return

        self.orchestrator = orchestrator
        self.command_fn = command_fn  # Callable to execute a command (from main)
        self.host = host
        self.port = port
        self._server_thread: Optional[threading.Thread] = None
        self._alerts_buffer = []  # Recent security alerts

        self.app = FastAPI(
            title="SentinelAI Dashboard",
            description="World-class personal AI assistant control panel",
            version="2.0.0"
        )

        # CORS for local tools / mobile companion
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._templates = None
        if os.path.isdir(TEMPLATE_DIR):
            self._templates = Jinja2Templates(directory=TEMPLATE_DIR)

        self._register_routes()
        logger.info(f"SentinelWebServer configured on {host}:{port}")

    def _register_routes(self):
        app = self.app

        # ── Dashboard UI ────────────────────────────────────────────────────
        @app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            stats = self._get_stats()
            if self._templates:
                return self._templates.TemplateResponse(
                    "index.html", {"request": request, "stats": stats}
                )
            return HTMLResponse("<h1>SentinelAI</h1><p>Template directory not found.</p>")

        # ── Command endpoint ─────────────────────────────────────────────────
        @app.post("/command")
        async def run_command(req: CommandRequest):
            cmd = (req.command or "").strip()
            if not cmd:
                raise HTTPException(status_code=400, detail="Empty command")

            response_text = ""
            try:
                if self.command_fn:
                    # Run command in thread, collect response
                    result_holder = {"response": ""}
                    event = threading.Event()

                    def _run():
                        try:
                            self.command_fn(cmd)
                            result_holder["response"] = f"Executed: {cmd}"
                        except Exception as e:
                            result_holder["response"] = f"Error: {e}"
                        finally:
                            event.set()

                    t = threading.Thread(target=_run, daemon=True)
                    t.start()
                    event.wait(timeout=30)
                    response_text = result_holder["response"]

                elif self.orchestrator:
                    response_text = self.orchestrator.run_command(cmd) or "Done."
                else:
                    response_text = "No command executor configured."
            except Exception as e:
                logger.error(f"Command error: {e}")
                response_text = f"Error: {e}"

            return {"response": response_text, "command": cmd, "timestamp": time.time()}

        # ── Stats endpoint ────────────────────────────────────────────────────
        @app.get("/stats")
        async def get_stats():
            stats = self._get_stats()
            # Drain alerts buffer
            alerts = list(self._alerts_buffer)
            self._alerts_buffer.clear()
            stats["alerts"] = alerts
            return stats

        # ── Conversation history ──────────────────────────────────────────────
        @app.get("/history")
        async def get_history():
            try:
                from sentinel.core.conversation import get_conversation
                conv = get_conversation()
                turns = [
                    {
                        "role": t.role,
                        "content": t.content,
                        "timestamp": t.timestamp,
                        "emotion": t.emotion
                    }
                    for t in list(conv._history)
                ]
                return {"turns": turns, "count": len(turns)}
            except Exception as e:
                return {"turns": [], "count": 0, "error": str(e)}

        # ── Generative UI / Dynamic Dashboard (Sci-Fi Tier) ────────────────────
        class UIGeneratorRequest(BaseModel):
            prompt: str

        @app.post("/generate-ui")
        async def generate_ui(req: UIGeneratorRequest):
            """Instantly generates functional React/Tailwind/HTML widgets based on prompt."""
            if not self.orchestrator:
                return JSONResponse({"html": "Orchestrator unavailable."}, status_code=500)
            
            sys_prompt = (
                "You are an expert Frontend Developer. The user is asking for a widget inside an AI dashboard. "
                "Output ONLY a raw, fully functional block of HTML containing embedded CSS/Tailwind (from CDN) and "
                "vanilla JS to satisfy the request. DO NOT output markdown ticks or explanation. "
                f"Request: {req.prompt}"
            )
            
            try:
                html_code = self.orchestrator._safe_llm_call(sys_prompt)
                
                # Clean up wrapping markdown if the model hallucinates it
                if "```html" in html_code:
                    html_code = html_code.split("```html")[1].split("```")[0].strip()
                elif "```" in html_code:
                    html_code = html_code.split("```")[1].split("```")[0].strip()
                    
                return JSONResponse({"html": html_code})
            except Exception as e:
                logger.error(f"Generative UI failed: {e}")
                return JSONResponse({"html": f"<div style='color:red;'>UI Gen Failed: {e}</div>"})

        # ── Knowledge base endpoints ──────────────────────────────────────────
        @app.post("/kb/add")
        async def kb_add(req: KBAddRequest):
            try:
                if self.orchestrator and self.orchestrator.knowledge_base:
                    ok, msg = self.orchestrator.knowledge_base.add_information(
                        req.text, metadata={"source": req.source}
                    )
                    return {"success": ok, "message": msg}
                return {"success": False, "message": "Knowledge base not available"}
            except Exception as e:
                return {"success": False, "message": str(e)}

        @app.post("/kb/query")
        async def kb_query(req: KBQueryRequest):
            try:
                if self.orchestrator and self.orchestrator.knowledge_base:
                    result = self.orchestrator.knowledge_base.query_knowledge(
                        req.query, n_results=req.n_results
                    )
                    return {"result": result, "query": req.query}
                return {"result": "Knowledge base not available", "query": req.query}
            except Exception as e:
                return {"result": str(e), "query": req.query}

        # ── Agent control endpoints ───────────────────────────────────────────
        @app.post("/agent/start")
        async def agent_start(req: CommandRequest):
            try:
                if self.orchestrator and self.orchestrator.agentic_engine:
                    goal = req.command or "Assist with my current task"
                    threading.Thread(
                        target=self.orchestrator.agentic_engine.execute_autonomous_goal,
                        args=(goal,),
                        daemon=True
                    ).start()
                    return {"status": "started", "goal": goal}
                return {"status": "error", "message": "Agentic engine not available"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        @app.post("/agent/stop")
        async def agent_stop():
            try:
                if self.orchestrator and self.orchestrator.agentic_engine:
                    self.orchestrator.agentic_engine.stop()
                    return {"status": "stopped"}
                return {"status": "error", "message": "Agentic engine not available"}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        # ── Health check ──────────────────────────────────────────────────────
        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "version": "2.0.0",
                "uptime": time.time(),
                "modules": self._get_module_health()
            }

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _get_stats(self) -> Dict[str, Any]:
        """Collect system stats from all modules."""
        stats = {
            "porcupine_status": "Active",
            "gemini_status": "Active" if os.getenv("GEMINI_API_KEY") else "No Key",
            "ollama_status": "Unknown",
            "cpu_temp": None,
            "gpu_temp": None,
            "kb_count": 0,
        }
        if self.orchestrator:
            try:
                stats["cpu_temp"] = self.orchestrator.system_monitor.get_cpu_temp()
            except Exception:
                pass
            try:
                stats["gpu_temp"] = self.orchestrator.system_monitor.get_gpu_temp()
            except Exception:
                pass
            try:
                stats["ollama_status"] = "Active" if self.orchestrator.ollama_module.is_available() else "Offline"
            except Exception:
                pass
            try:
                stats["kb_count"] = self.orchestrator.knowledge_base.get_count()
            except Exception:
                pass
        return stats

    def _get_module_health(self) -> Dict[str, bool]:
        """Returns online/offline status for each module."""
        if not self.orchestrator:
            return {}
        modules = {
            "vision": "vision_module",
            "knowledge_base": "knowledge_base",
            "emotion": "emotion_module",
            "habits": "habit_learner",
            "security": "security_shield",
            "iot": "iot_hub",
            "coding": "coding_module",
            "automation": "automation_module",
            "gesture": "gesture_module",
            "file_intel": "file_intel",
            "calendar": "calendar_module",
            "web_agent": "web_agent",
            "playwright": "playwright_agent",
            "agentic_engine": "agentic_engine",
            "daily_brief": "daily_brief",
            "ollama": "ollama_module",
        }
        health = {}
        for label, attr in modules.items():
            health[label] = getattr(self.orchestrator, attr, None) is not None
        return health

    def add_alert(self, message: str):
        """Add a security alert to the alerts buffer for the next /stats poll."""
        self._alerts_buffer.append(message)
        if len(self._alerts_buffer) > 20:
            self._alerts_buffer = self._alerts_buffer[-20:]

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        """Start the web server in a background daemon thread."""
        if not _FASTAPI_AVAILABLE:
            return
        self._server_thread = threading.Thread(
            target=self._run_uvicorn, daemon=True
        )
        self._server_thread.start()
        logger.info(f"Web dashboard started at http://{self.host}:{self.port}")

    def _run_uvicorn(self):
        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="warning",
                access_log=False
            )
        except Exception as e:
            logger.error(f"Web server error: {e}")
