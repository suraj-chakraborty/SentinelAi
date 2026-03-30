from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import threading
import logging
import os
from datetime import datetime

class SentinelWebServer:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.app = FastAPI(title="SentinelAI Dashboard")
        self.logger = logging.getLogger("SentinelWebServer")
        
        # Setup templates
        self.templates_path = os.path.join(os.path.dirname(__file__), "..", "templates")
        os.makedirs(self.templates_path, exist_ok=True)
        self.templates = Jinja2Templates(directory=self.templates_path)
        
        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            stats = {
                "cpu_temp": self.orchestrator.system_monitor.get_cpu_temp(),
                "gpu_temp": self.orchestrator.system_monitor.get_gpu_temp(),
                "last_active": datetime.now().strftime("%H:%M:%S"),
                "ollama_status": "Online" if self.orchestrator.ollama_module.is_available() else "Offline",
                "knowledge_count": self.orchestrator.knowledge_base.collection.count(),
                "porcupine_status": "Active" if os.getenv("PVPORCUPINE_PRIVATE_KEY") else "Missing Key",
                "gemini_status": "Active" if os.getenv("GEMINI_API_KEY") else "Missing Key"
            }
            return self.templates.TemplateResponse("index.html", {"request": request, "stats": stats})

        @self.app.post("/command")
        async def handle_command(request: Request):
            data = await request.json()
            command = data.get("command")
            if not command:
                return JSONResponse({"status": "error", "message": "No command provided."})
            
            self.logger.info(f"Web command: {command}")
            response = self.orchestrator.run_command(command)
            return JSONResponse({"status": "success", "response": response})

        @self.app.get("/stats")
        async def get_stats():
            return JSONResponse({
                "cpu": self.orchestrator.system_monitor.get_cpu_temp(),
                "gpu": self.orchestrator.system_monitor.get_gpu_temp(),
                "memory": self.orchestrator.knowledge_base.collection.count(),
                "alerts": self.orchestrator.check_periodic_alerts()
            })

    def start(self, host="127.0.0.1", port=8000):
        """Starts the web server in a background thread."""
        def _run():
            uvicorn.run(self.app, host=host, port=port, log_level="error")
        
        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()
        self.logger.info(f"Sentinel Web Dashboard started at http://{host}:{port}")
