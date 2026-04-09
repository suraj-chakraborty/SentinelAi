from __future__ import annotations
import json
import sqlite3
import os
from typing import Optional, List

from sentinel.models.user_profile import UserProfile


class UserProfileManager:
    """SQLite-backed per-user profile store (Phase 5.1).

    - Default DB path: ~/AppData/Roaming/SentinelAi/user_profiles.sqlite3 (Windows-friendly)
    - SQLite schema:
      user_profiles(
        user_id TEXT PRIMARY KEY,
        persona TEXT,
        memory_ttl_hours INTEGER,
        opt_in_privacy INTEGER,
        preferred_tools TEXT
      )
    - preferred_tools stored as JSON array string
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            # Windows-friendly default path
            home = os.path.expanduser("~")
            db_path = os.path.join(home, "AppData", "Roaming", "SentinelAi", "user_profiles.sqlite3")
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
              user_id TEXT PRIMARY KEY,
              persona TEXT,
              memory_ttl_hours INTEGER,
              opt_in_privacy INTEGER,
              preferred_tools TEXT
            )
            """
        )
        self._conn.commit()

    def _to_profile(self, row: sqlite3.Row) -> UserProfile:
        return UserProfile(
            user_id=row["user_id"],
            persona=row["persona"],
            memory_ttl_hours=int(row["memory_ttl_hours"] or 0),
            opt_in_privacy=bool(row["opt_in_privacy"]),
            preferred_tools=json.loads(row["preferred_tools"] or "[]"),
        )

    def set_profile(self, user_id: str, persona: str = None, memory_ttl_hours: int = 0,
                    opt_in_privacy: bool = True, preferred_tools=None) -> UserProfile:
        if preferred_tools is None:
            preferred_tools = []
        row = (user_id, persona if persona is not None else "calm Jarvis",
               int(memory_ttl_hours), int(bool(opt_in_privacy)), json.dumps(preferred_tools))
        self._conn.execute(
            "INSERT OR REPLACE INTO user_profiles (user_id, persona, memory_ttl_hours, opt_in_privacy, preferred_tools) VALUES (?, ?, ?, ?, ?)",
            row,
        )
        self._conn.commit()
        return self.get_profile(user_id)

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        cur = self._conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        if not row:
            return None
        return self._to_profile(row)

    def update_profile(self, user_id: str, **updates) -> Optional[UserProfile]:
        p = self.get_profile(user_id)
        if not p:
            return None
        data = p.to_dict()
        data.update(updates)
        return self.set_profile(
            user_id,
            persona=data.get('persona'),
            memory_ttl_hours=data.get('memory_ttl_hours', 0),
            opt_in_privacy=data.get('opt_in_privacy', True),
            preferrred_tools=data.get('preferred_tools', [])
        )

    def delete_profile(self, user_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
        self._conn.commit()
        return cur.rowcount > 0
