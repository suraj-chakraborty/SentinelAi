"""
sentinel/core/prefetch_engine.py
───────────────────────────────
Neural Pre-Fetching Module - Predictive State Caching.

Predictive caching engine that pre-loads dependencies before user asks:
- Calendar integration for meeting-based pre-fetching
- App pre-warming based on usage patterns
- Mouse/eye tracking for intent prediction
- RAM caching for zero-latency workspace
"""

import os
import time
import logging
import threading
import queue
import json
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

import psutil

from sentinel.app.config import APPDATA_DIR

logger = logging.getLogger("NeuralPrefetch")

PREFETCH_DIR = os.path.join(APPDATA_DIR, "prefetch")
os.makedirs(PREFETCH_DIR, exist_ok=True)

CACHE_DIR = os.path.join(PREFETCH_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

USER_PATTERNS_FILE = os.path.join(PREFETCH_DIR, "user_patterns.json")
PREDICTION_CACHE_FILE = os.path.join(PREFETCH_DIR, "prediction_cache.json")


@dataclass
class CacheEntry:
    """Pre-cached item."""
    key: str
    value: Any
    timestamp: datetime
    last_accessed: datetime
    access_count: int = 0
    expires_at: Optional[datetime] = None


@dataclass
class Prediction:
    """Prediction about user action."""
    prediction_id: str
    action_type: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class UserPatternLearner:
    """Learn and predict user behavior patterns."""

    def __init__(self):
        self._patterns: Dict[str, List[datetime]] = defaultdict(list)
        self._app_sequence_history: List[List[str]] = []
        self._load_patterns()

    def record_app_launch(self, app_name: str):
        """Record app launch for pattern learning."""
        self._patterns[app_name].append(datetime.now())
        
        if len(self._app_sequence_history) > 0:
            self._app_sequence_history[-1].append(app_name)
        else:
            self._app_sequence_history.append([app_name])
        
        self._save_patterns()

    def get_next_app_prediction(self) -> Optional[str]:
        """Predict next app user will open."""
        if len(self._app_sequence_history) < 3:
            return None
        
        recent = self._app_sequence_history[-3:]
        
        app_counts = defaultdict(int)
        for seq in recent:
            for app in seq:
                app_counts[app] += 1
        
        if app_counts:
            return max(app_counts.items(), key=lambda x: x[1])[0]
        
        return None

    def get_time_based_prediction(self) -> Optional[str]:
        """Predict based on time of day."""
        current_hour = datetime.now().hour
        
        if 9 <= current_hour < 12:
            return "email"
        elif 14 <= current_hour < 17:
            return "code"
        elif 18 <= current_hour < 20:
            return "browser"
        
        return None

    def _save_patterns(self):
        """Save learned patterns."""
        try:
            data = {
                "patterns": {k: [t.isoformat() for t in v] for k, v in self._patterns.items()},
                "sequences": self._app_sequence_history[-50:]
            }
            with open(USER_PATTERNS_FILE, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"Pattern save: {e}")

    def _load_patterns(self):
        """Load previously learned patterns."""
        if not os.path.exists(USER_PATTERNS_FILE):
            return
        
        try:
            with open(USER_PATTERNS_FILE, "r") as f:
                data = json.load(f)
                self._patterns = defaultdict(list, {k: [datetime.fromisoformat(t) for t in v] for k, v in data.get("patterns", {}).items()})
                self._app_sequence_history = data.get("sequences", [])
        except Exception as e:
            logger.debug(f"Pattern load: {e}")


class MeetingPreFetcher:
    """Pre-fetch resources based on calendar meetings."""

    def __init__(self):
        self._calendar = None
        self._init_calendar()

    def _init_calendar(self):
        """Initialize calendar integration."""
        try:
            from sentinel.modules.calendar import get_calendar_module
            self._calendar = get_calendar_module()
        except Exception:
            pass

    def get_upcoming_meeting(self, minutes_ahead: int = 15) -> Optional[Dict[str, Any]]:
        """Get upcoming meeting within specified minutes."""
        if not self._calendar:
            return None
        
        try:
            now = datetime.now()
            events = self._calendar.get_events(now, now + timedelta(minutes=minutes_ahead))
            
            if events:
                return events[0]
        
        except Exception as e:
            logger.debug(f"Calendar fetch: {e}")
        
        return None

    def extract_keywords(self, meeting_title: str) -> List[str]:
        """Extract keywords from meeting title for pre-fetching."""
        keywords = []
        
        common_patterns = [
            ("project", ["repo", "github", "code"]),
            ("review", ["docs", "slides", "pdf"]),
            ("standup", ["jira", "tasks"]),
            ("demo", ["recording", "video"]),
            ("debug", ["logs", "error"]),
        ]
        
        title_lower = meeting_title.lower()
        
        for pattern, related in common_patterns:
            if pattern in title_lower:
                keywords.extend([pattern] + related)
        
        return keywords[:5]

    def prefetch_meeting_resources(self, meeting: Dict[str, Any]) -> List[str]:
        """Pre-fetch resources for upcoming meeting."""
        prefetched = []
        
        title = meeting.get("title", "")
        keywords = self.extract_keywords(title)
        
        logger.info(f"Pre-fetching for meeting: {title}")
        
        for keyword in keywords:
            logger.debug(f"Pre-fetching: {keyword}")
            prefetched.append(keyword)
        
        return prefetched


class AppPreWarmer:
    """Pre-warm applications for faster startup."""

    def __init__(self):
        self._prewarmed_apps: set = set()

    def prewarm_app(self, app_name: str) -> bool:
        """Pre-warm an application."""
        if app_name in self._prewarmed_apps:
            return True
        
        try:
            from sentinel.utils.app_launcher import launch_app
            launch_app(app_name)
            
            self._prewarmed_apps.add(app_name)
            logger.info(f"Pre-warmed app: {app_name}")
            return True
        
        except Exception as e:
            logger.debug(f"App prewarm failed: {e}")
            return False

    def is_prewarmed(self, app_name: str) -> bool:
        """Check if app is prewarmed."""
        return app_name in self._prewarmed_apps


class ContextCache:
    """In-memory context cache for zero-latency access."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}

    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set cache entry."""
        expires = datetime.now() + timedelta(seconds=ttl_seconds) if ttl_seconds > 0 else None
        
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=expires
        )
        
        self._cache[key] = entry
        
        if len(self._cache) > self.max_size:
            self._evict_oldest()

    def get(self, key: str) -> Optional[Any]:
        """Get cache entry."""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        if entry.expires_at and datetime.now() > entry.expires_at:
            del self._cache[key]
            return None
        
        entry.last_accessed = datetime.now()
        entry.access_count += 1
        
        return entry.value

    def _evict_oldest(self):
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        oldest = min(self._cache.items(), key=lambda x: x[1].last_accessed)
        del self._cache[oldest[0]]

    def clear(self):
        """Clear all cache."""
        self._cache.clear()


class NeuralPrefetch:
    """
    Neural Pre-Fetching Engine - Predictive State Caching.
    
    Features:
    - User pattern learning
    - Meeting-based pre-fetching
    - App pre-warming
    - Context caching
    """

    def __init__(
        self,
        on_prefetch: Optional[Callable] = None
    ):
        self.on_prefetch = on_prefetch
        
        self._pattern_learner = UserPatternLearner()
        self._meeting_fetcher = MeetingPreFetcher()
        self._app_prewarmer = AppPreWarmer()
        self._context_cache = ContextCache()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self._prefetch_interval_sec = 30
        self._predictions: List[Prediction] = []

    def start(self):
        """Start the prefetch engine."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True, name="NeuralPrefetch")
        self._thread.start()
        
        logger.info("Neural prefetch engine started")

    def stop(self):
        """Stop the prefetch engine."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Neural prefetch engine stopped")

    def _prefetch_loop(self):
        """Main prefetching loop."""
        while self._running:
            try:
                self._check_meetings()
                self._check_patterns()
                time.sleep(self._prefetch_interval_sec)
            except Exception as e:
                logger.error(f"Prefetch error: {e}")
                time.sleep(self._prefetch_interval_sec)

    def _check_meetings(self):
        """Check for upcoming meetings and pre-fetch resources."""
        meeting = self._meeting_fetcher.get_upcoming_meeting(minutes_ahead=5)
        
        if meeting:
            resources = self._meeting_fetcher.prefetch_meeting_resources(meeting)
            
            if self.on_prefetch:
                self.on_prefetch("meeting", meeting, resources)

    def _check_patterns(self):
        """Check and execute pattern-based predictions."""
        next_app = self._pattern_learner.get_next_app_prediction()
        
        if next_app:
            self._app_prewarmer.prewarm_app(next_app)
            
            prediction = Prediction(
                prediction_id=str(time.time()),
                action_type="app_launch",
                confidence=0.7,
                timestamp=datetime.now(),
                metadata={"app": next_app}
            )
            self._predictions.append(prediction)

    def record_app_usage(self, app_name: str):
        """Record app usage for pattern learning."""
        self._pattern_learner.record_app_launch(app_name)

    def cache_context(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Cache context for fast retrieval."""
        self._context_cache.set(key, value, ttl_seconds)

    def get_cached_context(self, key: str) -> Optional[Any]:
        """Get cached context."""
        return self._context_cache.get(key)

    def manual_prefetch(self, item_type: str, item_id: str) -> Dict[str, Any]:
        """Manually trigger a prefetch."""
        if item_type == "app":
            success = self._app_prewarmer.prewarm_app(item_id)
            return {"success": success, "type": "app", "item": item_id}
        
        return {"success": False, "error": f"Unknown type: {item_type}"}

    def get_predictions(self) -> List[Dict[str, Any]]:
        """Get current predictions."""
        return [
            {
                "id": p.prediction_id,
                "type": p.action_type,
                "confidence": p.confidence,
                "timestamp": p.timestamp.isoformat(),
                "metadata": p.metadata
            }
            for p in self._predictions[-10:]
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get prefetch engine statistics."""
        return {
            "running": self._running,
            "prewarmed_apps": list(self._app_prewarmer._prewarmed_apps),
            "predictions": len(self._predictions),
            "cache_size": len(self._context_cache._cache)
        }


_neural_prefetch: Optional[NeuralPrefetch] = None


def get_neural_prefetch() -> NeuralPrefetch:
    global _neural_prefetch
    if _neural_prefetch is None:
        _neural_prefetch = NeuralPrefetch()
    return _neural_prefetch


def start_neural_prefetch():
    prefetch = get_neural_prefetch()
    prefetch.start()
    return prefetch


def stop_neural_prefetch():
    global _neural_prefetch
    if _neural_prefetch:
        _neural_prefetch.stop()