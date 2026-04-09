"""
sentinel/commands/search_memory.py
───────────────────────────────────
Command to search semantic screen memory.
"""

import logging
from typing import Dict, Any, Optional

from sentinel.memory.screen_indexer import get_screen_indexer, start_screen_indexer, stop_screen_indexer
from sentinel.memory.screen_memory_db import get_screen_memory_db

logger = logging.getLogger(__name__)


class SearchMemoryCommand:
    def __init__(self):
        self._indexer = None

    def execute(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        try:
            indexer = get_screen_indexer()
            results = indexer.search(query, n_results)
            
            if not results:
                return {
                    "success": True,
                    "query": query,
                    "results": [],
                    "message": "No matching screens found"
                }
            
            formatted = []
            for r in results:
                formatted.append({
                    "screen_id": r["id"],
                    "text_preview": r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"],
                    "timestamp": r["metadata"].get("timestamp"),
                    "screenshot": r["metadata"].get("screenshot_path"),
                    "distance": r.get("distance")
                })
            
            return {
                "success": True,
                "query": query,
                "results": formatted,
                "count": len(formatted)
            }
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return {"success": False, "error": str(e)}

    def get_recent(self, limit: int = 10) -> Dict[str, Any]:
        try:
            indexer = get_screen_indexer()
            screens = indexer.get_recent(limit)
            
            formatted = []
            for s in screens:
                formatted.append({
                    "screen_id": s["id"],
                    "text_preview": s["text"][:300] + "..." if len(s["text"]) > 300 else s["text"],
                    "timestamp": s["metadata"].get("timestamp"),
                    "screenshot": s["metadata"].get("screenshot_path")
                })
            
            return {
                "success": True,
                "screens": formatted,
                "count": len(formatted)
            }
        except Exception as e:
            logger.error(f"Get recent screens failed: {e}")
            return {"success": False, "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        try:
            indexer = get_screen_indexer()
            return indexer.get_stats()
        except Exception as e:
            logger.error(f"Get stats failed: {e}")
            return {"success": False, "error": str(e)}

    def start_indexing(self, interval_seconds: int = 60) -> Dict[str, Any]:
        try:
            start_screen_indexer(interval_seconds)
            return {"success": True, "message": f"Screen indexing started (interval: {interval_seconds}s)"}
        except Exception as e:
            logger.error(f"Start indexing failed: {e}")
            return {"success": False, "error": str(e)}

    def stop_indexing(self) -> Dict[str, Any]:
        try:
            stop_screen_indexer()
            return {"success": True, "message": "Screen indexing stopped"}
        except Exception as e:
            logger.error(f"Stop indexing failed: {e}")
            return {"success": False, "error": str(e)}

    def capture_now(self) -> Dict[str, Any]:
        try:
            indexer = get_screen_indexer()
            screen_id = indexer.index_now()
            if screen_id:
                return {"success": True, "screen_id": screen_id, "message": "Screen captured and indexed"}
            return {"success": False, "error": "No text extracted from screen"}
        except Exception as e:
            logger.error(f"Capture now failed: {e}")
            return {"success": False, "error": str(e)}

    def delete_screen(self, screen_id: str) -> Dict[str, Any]:
        try:
            db = get_screen_memory_db()
            if db.delete_screen(screen_id):
                return {"success": True, "message": f"Screen {screen_id} deleted"}
            return {"success": False, "error": "Failed to delete screen"}
        except Exception as e:
            logger.error(f"Delete screen failed: {e}")
            return {"success": False, "error": str(e)}


_search_memory_command: Optional[SearchMemoryCommand] = None


def get_search_memory_command() -> SearchMemoryCommand:
    global _search_memory_command
    if _search_memory_command is None:
        _search_memory_command = SearchMemoryCommand()
    return _search_memory_command