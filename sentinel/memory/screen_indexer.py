"""
sentinel/memory/screen_indexer.py
──────────────────────────────────
Background daemon that periodically captures screens and indexes them
into the semantic screen memory for recall.
"""

import os
import time
import logging
import threading
import queue
from datetime import datetime
from typing import Optional, Callable

from sentinel.memory.screen_capture import ScreenCapture
from sentinel.memory.screen_memory_db import get_screen_memory_db

logger = logging.getLogger(__name__)


class ScreenIndexer:
    def __init__(
        self,
        interval_seconds: int = 60,
        capture_enabled: bool = True,
        on_index: Optional[Callable] = None
    ):
        self.interval_seconds = interval_seconds
        self.capture_enabled = capture_enabled
        self.on_index = on_index
        
        self._capture = ScreenCapture()
        self._db = get_screen_memory_db()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._event_queue = queue.Queue()

    def start(self):
        if self._running:
            logger.warning("Screen indexer already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ScreenIndexer")
        self._thread.start()
        logger.info(f"Screen indexer started (interval: {self.interval_seconds}s)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Screen indexer stopped")

    def _run_loop(self):
        while self._running:
            try:
                if self.capture_enabled:
                    self._capture_and_index()
                time.sleep(self.interval_seconds)
            except Exception as e:
                logger.error(f"Screen indexer loop error: {e}")
                time.sleep(self.interval_seconds)

    def _capture_and_index(self):
        try:
            result = self._capture.capture_and_index()
            if result:
                text, filepath = result
                screen_id = self._db.add_screen_memory(
                    text=text,
                    screenshot_path=filepath,
                    metadata={"source": "auto_index"}
                )
                
                self._event_queue.put({
                    "type": "indexed",
                    "screen_id": screen_id,
                    "timestamp": datetime.now().isoformat(),
                    "text_preview": text[:200] + "..." if len(text) > 200 else text
                })
                
                if self.on_index:
                    self.on_index(screen_id, text)
                
                logger.info(f"Indexed screen: {screen_id}")
        except Exception as e:
            logger.error(f"Failed to capture and index: {e}")

    def index_now(self) -> Optional[str]:
        result = self._capture.capture_and_index()
        if result:
            text, filepath = result
            return self._db.add_screen_memory(
                text=text,
                screenshot_path=filepath,
                metadata={"source": "manual"}
            )
        return None

    def search(self, query: str, n_results: int = 5):
        return self._db.search(query, n_results)

    def get_recent(self, limit: int = 10):
        return self._db.get_recent_screens(limit)

    def get_stats(self):
        return self._db.get_stats()

    def get_events(self, timeout: float = 0.1):
        events = []
        while True:
            try:
                event = self._event_queue.get_nowait()
                events.append(event)
            except queue.Empty:
                break
        return events


_screen_indexer: Optional[ScreenIndexer] = None


def get_screen_indexer() -> ScreenIndexer:
    global _screen_indexer
    if _screen_indexer is None:
        _screen_indexer = ScreenIndexer()
    return _screen_indexer


def start_screen_indexer(interval_seconds: int = 60):
    indexer = get_screen_indexer()
    indexer.interval_seconds = interval_seconds
    indexer.start()
    return indexer


def stop_screen_indexer():
    global _screen_indexer
    if _screen_indexer:
        _screen_indexer.stop()