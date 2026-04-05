"""
sentinel/memory/long_term_memory.py
─────────────────────────────────────
Cross-session persistent memory for SentinelAI.

Architecture
────────────
  • Short-term  : ConversationManager (rolling 20-turn window in RAM)
  • Long-term   : This module — ChromaDB-backed semantic store that persists
                  session summaries and important facts indefinitely.

How it works
────────────
  1. At session end (or every 20 turns), `summarize_session()` is called.
     The LLM condenses the conversation into 3-5 bullet-point facts.
  2. The summary is embedded and stored in a dedicated ChromaDB collection.
  3. On each new session, `inject_past_context()` retrieves the 3 most
     semantically relevant past summaries and prepends them to the system
     prompt so Sentinel "remembers" past interactions.
  4. Explicit facts can also be stored/queried independently via
     `store_fact()` / `recall_facts()`.

Usage
─────
    ltm = LongTermMemory(llm_callback=orchestrator._safe_llm_call)
    ltm.summarize_session(conversation_manager)         # call at session end
    context = ltm.inject_past_context("email project")  # called at session start
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import List, Optional

logger = logging.getLogger("LongTermMemory")

APPDATA_DIR = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi")
_LTM_DB_PATH = os.path.join(APPDATA_DIR, "ltm_db")


class LongTermMemory:
    """
    ChromaDB-backed long-term semantic memory.

    All heavy imports (chromadb, sentence-transformers) are done lazily
    so the class can be imported even in environments where those packages
    are unavailable — it degrades gracefully.
    """

    _SUMMARY_COLLECTION = "session_summaries"
    _FACTS_COLLECTION   = "user_facts"
    _MAX_CONTEXT_ITEMS  = 3       # how many past summaries to inject
    _SUMMARY_INTERVAL   = 20      # turns before auto-summarise

    def __init__(
        self,
        llm_callback=None,
        db_path: str = _LTM_DB_PATH,
    ):
        self.llm_callback = llm_callback
        self.db_path = db_path
        self._client = None
        self._summaries_col = None
        self._facts_col = None
        self._turns_since_last_summary = 0
        self._init_db()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            os.makedirs(self.db_path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.db_path)
            ef = embedding_functions.DefaultEmbeddingFunction()

            self._summaries_col = self._client.get_or_create_collection(
                name=self._SUMMARY_COLLECTION,
                embedding_function=ef,
            )
            self._facts_col = self._client.get_or_create_collection(
                name=self._FACTS_COLLECTION,
                embedding_function=ef,
            )
            logger.info(
                "Long-term memory ready (%d summaries, %d facts).",
                self._summaries_col.count(),
                self._facts_col.count(),
            )
        except ImportError:
            logger.warning("chromadb not installed — long-term memory disabled.")
        except Exception as exc:
            logger.error("LTM DB init failed: %s", exc)

    @property
    def _available(self) -> bool:
        return self._client is not None

    # ── Session summarisation ─────────────────────────────────────────────────

    def on_new_turn(self, conversation_manager) -> None:
        """
        Call after every new conversation turn.
        Triggers auto-summarisation every `_SUMMARY_INTERVAL` turns.
        """
        self._turns_since_last_summary += 1
        if self._turns_since_last_summary >= self._SUMMARY_INTERVAL:
            self.summarize_session(conversation_manager)
            self._turns_since_last_summary = 0

    def summarize_session(self, conversation_manager) -> Optional[str]:
        """
        Ask the LLM to condense recent conversation into bullet-point facts,
        then store in ChromaDB.
        Returns the summary text, or None on failure.
        """
        if not self._available or not self.llm_callback:
            return None

        history_text = conversation_manager.get_recent_context(n=20)
        if not history_text.strip():
            return None

        prompt = (
            "You are a memory distillation assistant. Summarise the following "
            "conversation into 3-5 concise bullet points capturing the key facts, "
            "decisions, and preferences mentioned. Be specific and factual.\n\n"
            f"Conversation:\n{history_text}\n\n"
            "Summary bullets:"
        )
        try:
            summary = self.llm_callback(prompt)
            if not summary or len(summary) < 10:
                return None

            doc_id = f"session_{uuid.uuid4().hex}"
            self._summaries_col.add(
                documents=[summary],
                metadatas=[{
                    "timestamp": time.time(),
                    "date": time.strftime("%Y-%m-%d %H:%M"),
                    "turns": self._turns_since_last_summary,
                }],
                ids=[doc_id],
            )
            logger.info("Session summary stored (id=%s).", doc_id)
            return summary
        except Exception as exc:
            logger.error("Failed to store session summary: %s", exc)
            return None

    # ── Context injection ─────────────────────────────────────────────────────

    def inject_past_context(self, current_topic: str = "") -> str:
        """
        Retrieve the most relevant past session summaries and format
        them as a system prompt prefix.

        Args:
            current_topic: The user's opening query / topic to guide retrieval.

        Returns:
            A formatted string of past context, or empty string if unavailable.
        """
        if not self._available:
            return ""
        try:
            count = self._summaries_col.count()
            if count == 0:
                return ""

            query = current_topic or "What has Sentinel helped me with recently?"
            n = min(self._MAX_CONTEXT_ITEMS, count)
            results = self._summaries_col.query(
                query_texts=[query],
                n_results=n,
                include=["documents", "metadatas"],
            )
            docs = (results.get("documents") or [[]])[0]
            metas = (results.get("metadatas") or [[]])[0]

            if not docs:
                return ""

            lines = ["[Past session memory]"]
            for doc, meta in zip(docs, metas):
                date = (meta or {}).get("date", "earlier")
                lines.append(f"— {date}: {doc.strip()}")

            return "\n".join(lines)
        except Exception as exc:
            logger.error("Failed to retrieve past context: %s", exc)
            return ""

    # ── Explicit fact storage ─────────────────────────────────────────────────

    def store_fact(self, fact: str, category: str = "general") -> bool:
        """Persist a single explicit fact about the user or their preferences."""
        if not self._available or not fact.strip():
            return False
        try:
            self._facts_col.add(
                documents=[fact],
                metadatas=[{"timestamp": time.time(), "category": category}],
                ids=[f"fact_{uuid.uuid4().hex}"],
            )
            logger.info("Fact stored: %s", fact[:60])
            return True
        except Exception as exc:
            logger.error("Failed to store fact: %s", exc)
            return False

    def recall_facts(self, query: str, n: int = 5) -> str:
        """Semantically retrieve stored facts related to a query."""
        if not self._available:
            return ""
        try:
            count = self._facts_col.count()
            if count == 0:
                return ""
            results = self._facts_col.query(
                query_texts=[query],
                n_results=min(n, count),
                include=["documents"],
            )
            docs = (results.get("documents") or [[]])[0]
            return "\n".join(f"• {d}" for d in docs) if docs else ""
        except Exception as exc:
            logger.error("Failed to recall facts: %s", exc)
            return ""

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        if not self._available:
            return {"available": False}
        try:
            return {
                "available": True,
                "session_summaries": self._summaries_col.count(),
                "stored_facts": self._facts_col.count(),
                "db_path": self.db_path,
            }
        except Exception as exc:
            return {"available": False, "error": str(exc)}
