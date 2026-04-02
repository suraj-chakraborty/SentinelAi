"""
sentinel/core/logger.py
────────────────────────
Structured Logging with Contextual Metadata — Tier 4 Production Hardening.

Outputs rich, parseable JSON logs and beautiful console logs.
"""

import logging
import sys
import json
import traceback
from datetime import datetime
import os

def _resolve_log_file():
    candidates = [
        os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi", "logs"),
        os.path.join(os.getcwd(), ".sentinel_logs"),
        os.environ.get("TEMP", os.getcwd()),
    ]
    for base in candidates:
        try:
            os.makedirs(base, exist_ok=True)
            probe = os.path.join(base, "sentinel_structured.log")
            with open(probe, "a", encoding="utf-8"):
                pass
            return probe
        except Exception:
            continue
    return None

JSON_LOG_FILE = _resolve_log_file()

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "file": record.pathname,
            "line": record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = traceback.format_exception(*record.exc_info)
        return json.dumps(log_obj)

class ConsoleFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[90m",    # Gray
        logging.INFO: "\033[96m",     # Cyan
        logging.WARNING: "\033[93m",  # Yellow
        logging.ERROR: "\033[91m",    # Red
        logging.CRITICAL: "\033[95m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        time_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        msg = f"{color}[{time_str}] {record.levelname:<7} [{record.name}] {record.getMessage()}{self.RESET}"
        if record.exc_info:
            msg += f"\n{color}{traceback.format_exc()}{self.RESET}"
        return msg

class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            super().emit(record)
        except Exception:
            try:
                self.acquire()
                self.stream = open(os.devnull, "w", encoding="utf-8")
            except Exception:
                pass
            finally:
                try:
                    self.release()
                except Exception:
                    pass

def setup_logging(level=logging.INFO):
    """Initializes structured JSON logging to file and colored text logging to console."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 1. Console Handler (Colored, readable)
    console_stream = getattr(sys, "__stdout__", None) or sys.stdout
    if console_stream and not getattr(console_stream, "closed", False):
        try:
            ch = SafeStreamHandler(console_stream)
            ch.setLevel(level)
            ch.setFormatter(ConsoleFormatter())
            root_logger.addHandler(ch)
        except Exception:
            pass

    # 2. File Handler (JSON, structured for ELK/Datadog/etc)
    if JSON_LOG_FILE:
        try:
            fh = logging.FileHandler(JSON_LOG_FILE, encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(JSONFormatter())
            root_logger.addHandler(fh)
        except Exception:
            pass

    try:
        logging.getLogger("SentinelLogger").info("Structured logging initialized.")
    except Exception:
        pass

# Auto-setup on import
setup_logging()
