"""
sentinel/commands/search_memory.py
───────────────────────────────────
Unified Episodic & Semantic Memory Search Command.
Handles cross-session recall ("What did we do yesterday?", "Recall past commands", "Search screens").
"""

import logging
from typing import Dict, Any, Optional, Union

from sentinel.memory.episodic_logger import get_episodic_logger
from sentinel.memory.screen_indexer import get_screen_indexer

logger = logging.getLogger("SearchMemory")


class SearchMemoryCommand:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    def execute(self, query: str, entity: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """
        Processes memory recall requests for both voice commands and direct queries.
        """
        q_lower = query.lower()

        # 1. Check for yesterday / past actions
        if "yesterday" in q_lower or "previous session" in q_lower:
            episodes = get_episodic_logger().get_past_actions(days_ago=1)
            if episodes:
                commands_summary = ", ".join([f"'{e['user_command']}'" for e in episodes[:4]])
                msg = f"Yesterday, you executed: {commands_summary}."
                return msg
            return "I don't have recorded commands from yesterday in episodic memory."

        # 2. Check for recent commands / history
        if any(k in q_lower for k in ("recent commands", "what did i do", "what did we do", "last command", "history")):
            episodes = get_episodic_logger().get_recent(limit=4)
            if episodes:
                lines = [f"- '{e['user_command']}' ({e['date_str'].split(' ')[1][:5]})" for e in episodes]
                return f"Your most recent commands were: {'; '.join(lines)}."
            return "No recent command history found."

        # 3. Keyword search in episodic memory
        clean_search = entity or query.replace("search memory for", "").replace("search memory", "").replace("recall", "").strip()
        episodes = get_episodic_logger().search_episodes(clean_search, limit=3)
        if episodes:
            match = episodes[0]
            return f"I recall that on {match['date_str']}, you ran '{match['user_command']}'. The result was: {match['response'][:150]}."

        # 4. Fallback to semantic screen indexer
        try:
            indexer = get_screen_indexer()
            results = indexer.search(clean_search, limit=3)
            if results:
                top = results[0]
                preview = top.get("text", "")[:180]
                return f"Found a matching screen capture containing: '{preview}'."
        except Exception as e:
            logger.debug(f"Screen indexer search: {e}")

        return f"I could not find past records matching '{clean_search}'."

    def get_recent(self, limit: int = 10) -> Dict[str, Any]:
        """API method for UI/dashboard."""
        try:
            screens = get_screen_indexer().get_recent(limit)
            return {"success": True, "screens": screens, "count": len(screens)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """API method for UI/dashboard."""
        try:
            return get_screen_indexer().get_stats()
        except Exception as e:
            return {"success": False, "error": str(e)}