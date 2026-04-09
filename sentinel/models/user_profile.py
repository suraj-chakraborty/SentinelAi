from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class UserProfile:
    user_id: str
    persona: str = "calm Jarvis"
    memory_ttl_hours: int = 0  # 0 means TTL disabled
    opt_in_privacy: bool = True
    preferred_tools: List[str] = None

    def __post_init__(self):
        if self.preferred_tools is None:
            self.preferred_tools = []

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "UserProfile":
        if not data:
            return None
        return UserProfile(
            user_id=data.get("user_id"),
            persona=data.get("persona", "calm Jarvis"),
            memory_ttl_hours=int(data.get("memory_ttl_hours", 0)),
            opt_in_privacy=bool(data.get("opt_in_privacy", True)),
            preferred_tools=data.get("preferred_tools", []),
        )
