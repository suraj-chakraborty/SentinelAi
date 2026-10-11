"""
sentinel/memory/episodic_logger.py
──────────────────────────────────
Persistent SQLite Episodic Memory Logger.
Maintains chronological history of user commands, actions, active apps, and responses.
Enables cross-session recall ("What did we do yesterday?", "What was that command?").
"""

import os
import sqlite3
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger("EpisodicLogger")

APPDATA_DIR = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi")
os.makedirs(APPDATA_DIR, exist_ok=True)
_DB_PATH = os.path.join(APPDATA_DIR, "episodic_memory.db")


class EpisodicLogger:
    """
    Chronological episode recorder and search engine.
    """

    def __init__(self, db_path: str = _DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS episodes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        date_str TEXT,
                        user_command TEXT,
                        intent TEXT,
                        response TEXT,
                        active_app TEXT,
                        active_window TEXT
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_intent ON episodes(intent)")
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize episodic DB: {e}")

    def log_turn(
        self,
        user_command: str,
        intent: str = "general",
        response: str = "",
        active_app: str = "",
        active_window: str = ""
    ):
        """Record an interaction turn."""
        if not user_command or not user_command.strip():
            return
        now = time.time()
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO episodes (timestamp, date_str, user_command, intent, response, active_app, active_window)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (now, date_str, user_command.strip(), intent, response[:1000] if response else "", active_app, active_window))
                conn.commit()
        except Exception as e:
            logger.debug(f"Failed to log episode: {e}")

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent interaction episodes."""
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT date_str, user_command, intent, response, active_app FROM episodes ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.debug(f"Failed to fetch recent episodes: {e}")
            return []

    def search_episodes(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search episodes by keywords in user command or response."""
        try:
            pattern = f"%{query.strip()}%"
            with self._get_conn() as conn:
                cursor = conn.execute("""
                    SELECT date_str, user_command, intent, response, active_app
                    FROM episodes
                    WHERE user_command LIKE ? OR response LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (pattern, pattern, limit))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.debug(f"Failed to search episodes: {e}")
            return []

    def get_past_actions(self, days_ago: int = 1) -> List[Dict[str, Any]]:
        """Retrieve commands executed on a previous day."""
        target_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        try:
            with self._get_conn() as conn:
                cursor = conn.execute("""
                    SELECT date_str, user_command, intent, response, active_app
                    FROM episodes
                    WHERE date_str LIKE ?
                    ORDER BY timestamp ASC
                """, (f"{target_date}%",))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.debug(f"Failed to get past actions: {e}")
            return []


# Singleton accessor
_episodic_logger: Optional[EpisodicLogger] = None

def get_episodic_logger() -> EpisodicLogger:
    global _episodic_logger
    if _episodic_logger is None:
        _episodic_logger = EpisodicLogger()
    return _episodic_logger
