"""Retrieval Module (Phase 3) – lightweight RAG-like augmentation.

This module provides a thin wrapper around the KnowledgeBase to retrieve
relevant contextual information for a given prompt.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("RetrievalModule")


class RetrievalModule:
    def __init__(self, knowledge_base=None, llm_callback=None):
        self.knowledge_base = knowledge_base
        self.llm_callback = llm_callback

    def retrieve(self, prompt: str) -> Optional[str]:
        """Return a retrieved contextual snippet for the given prompt, if available."""
        if not self.knowledge_base:
            return ""
        try:
            if hasattr(self.knowledge_base, "query_knowledge"):
                return self.knowledge_base.query_knowledge(prompt)
        except Exception as exc:
            logger.debug("RetrievalModule failed: %s", exc)
        return ""
