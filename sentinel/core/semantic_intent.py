"""
Semantic intent classification using sentence embeddings (sentence-transformers).

Used only when keyword + fuzzy routing return UNKNOWN — adds recall for paraphrases
without slowing every command (lazy model load).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("SemanticIntent")

# Minimum cosine similarity (normalized embeddings) to accept a semantic label.
_DEFAULT_MIN_SIM = 0.36


def _semantic_disabled() -> bool:
    v = (os.getenv("SENTINEL_SEMANTIC_ROUTING") or "").strip().lower()
    return v in ("0", "false", "no", "off")


PROTOTYPE_UTTERANCES: Dict[str, List[str]] = {
    "open_app": [
        "open chrome",
        "launch notepad",
        "start the calculator",
        "run vscode",
        "open my browser",
        "start microsoft edge",
        "open spotify",
        "launch discord",
        "open file explorer",
        "start outlook",
    ],
    "close_app": [
        "close chrome",
        "exit notepad",
        "quit discord",
        "stop spotify",
        "close the browser",
        "kill firefox",
        "terminate calculator",
        "close this application",
        "shut down spotify",
    ],
    "shutdown": [
        "shut down the computer",
        "turn off my pc",
        "power off the machine",
        "shutdown windows",
        "switch off computer",
    ],
    "restart": [
        "restart the computer",
        "reboot windows",
        "restart my pc",
        "reboot the machine",
    ],
    "system_status": [
        "how much memory is left",
        "cpu usage",
        "check ram",
        "system temperature",
        "is my disk full",
        "how hot is the gpu",
        "computer performance",
        "battery status laptop",
    ],
    "autonomous_agent": [
        "organize my downloads folder",
        "find all pdf files on desktop",
        "automate filling this form",
        "what is on my screen right now",
        "summarize this long page",
        "create a folder structure",
        "rename these photos",
        "install software for me",
        "search the web and compile results",
    ],
    "deactivate": [
        "exit sentinel",
        "quit the assistant",
        "close sentinel ai",
        "deactivate the ai",
        "stop listening sentinel",
    ],
}


class SemanticIntentClassifier:
    """
    Lazy-loads MiniLM and compares the utterance to per-intent prototype clusters
    (mean pooled embedding per intent).
    """

    def __init__(self, min_similarity: float = _DEFAULT_MIN_SIM):
        self.min_similarity = min_similarity
        self._lock = threading.Lock()
        self._model = None
        self._intent_ids: List[str] = []
        self._matrix = None  # (n_intents, dim) row-normalized

    def _ensure_loaded(self) -> bool:
        if _semantic_disabled():
            return False
        with self._lock:
            if self._matrix is not None:
                return True
            try:
                import numpy as np
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.info(
                    "Semantic routing skipped: install sentence-transformers (see requirements.txt)."
                )
                return False

            try:
                model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning("Semantic routing: could not load embedding model: %s", e)
                return False

            intent_ids: List[str] = []
            vectors: List = []
            for intent_key, phrases in PROTOTYPE_UTTERANCES.items():
                if not phrases:
                    continue
                emb = model.encode(phrases, convert_to_numpy=True, show_progress_bar=False)
                mean = emb.mean(axis=0)
                norm = np.linalg.norm(mean)
                if norm > 0:
                    mean = mean / norm
                intent_ids.append(intent_key)
                vectors.append(mean)

            if not vectors:
                return False

            self._model = model
            self._intent_ids = intent_ids
            self._matrix = np.stack(vectors, axis=0)
            logger.info("Semantic intent routing enabled (%s labels).", len(intent_ids))
            return True

    def is_available(self) -> bool:
        return self._ensure_loaded()

    def predict(self, command: str) -> Tuple[Optional[str], float]:
        """
        Returns (intent_value_str, score) e.g. ('open_app', 0.52), or (None, 0.0) if no match.
        intent_value_str matches Intent enum .value strings.
        """
        text = (command or "").strip()
        if not text or len(text) > 512:
            return None, 0.0

        if not self._ensure_loaded():
            return None, 0.0

        import numpy as np

        q = self._model.encode(
            [text], convert_to_numpy=True, show_progress_bar=False
        )[0]
        nq = np.linalg.norm(q)
        if nq > 0:
            q = q / nq

        sims = self._matrix @ q
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score < self.min_similarity:
            return None, score
        return self._intent_ids[idx], score


_default_classifier: Optional[SemanticIntentClassifier] = None


def get_semantic_classifier() -> SemanticIntentClassifier:
    global _default_classifier
    if _default_classifier is None:
        raw = (os.getenv("SENTINEL_SEMANTIC_MIN_SIM") or "").strip()
        min_sim = _DEFAULT_MIN_SIM
        if raw:
            try:
                min_sim = float(raw)
            except ValueError:
                pass
        _default_classifier = SemanticIntentClassifier(min_similarity=min_sim)
    return _default_classifier
