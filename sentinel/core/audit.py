from __future__ import annotations
import json
import time
import os
from sentinel.app.config import AUDIT_LOG_PATH

def log_audit(actor: str, action: str, details: str = "") -> None:
    entry = {
        "timestamp": int(time.time()),
        "actor": actor,
        "action": action,
        "details": details,
    }
    try:
        os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
