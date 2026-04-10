"""
sentinel/memory/screen_memory_db.py
───────────────────────────────────
ChromaDB wrapper for semantic screen memory storage.
Stores screenshot OCR text as vectors for semantic recall.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_OK = True
except ImportError:
    CHROMADB_OK = False
    logging.warning("chromadb not available - screen memory DB disabled")

from sentinel.app.config import APPDATA_DIR

logger = logging.getLogger(__name__)

SCREEN_MEMORY_DIR = os.path.join(APPDATA_DIR, "screen_memory")
Path(SCREEN_MEMORY_DIR).mkdir(parents=True, exist_ok=True)


class ScreenMemoryDB:
    def __init__(self, persist_directory: str = SCREEN_MEMORY_DIR):
        if not CHROMADB_OK:
            raise RuntimeError("ChromaDB not available")
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
        self._embedding_function = None
        self._init_db()

    def _init_db(self):
        if not CHROMADB_OK:
            return
        try:
            from sentence_transformers import SentenceTransformer
            
            self._embedding_function = SentenceTransformer('all-MiniLM-L6-v2')
            
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self._collection = self._client.get_or_create_collection(
                name="screen_memory",
                metadata={"description": "OCR text from captured screens"}
            )
            logger.info(f"Screen memory DB initialized at {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize screen memory DB: {e}")
            raise

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._embedding_function.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def add_screen_memory(
        self,
        text: str,
        screenshot_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        screen_id = f"screen_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        meta = metadata or {}
        meta["timestamp"] = datetime.now().isoformat()
        if screenshot_path:
            meta["screenshot_path"] = screenshot_path
        meta["text_length"] = len(text)
        
        try:
            self._collection.add(
                ids=[screen_id],
                documents=[text],
                metadatas=[meta]
            )
            logger.info(f"Added screen memory: {screen_id}")
            return screen_id
        except Exception as e:
            logger.error(f"Failed to add screen memory: {e}")
            raise

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        try:
            query_embedding = self._embed_texts([query])[0]
            
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            
            if not results or not results.get("ids"):
                return []
            
            formatted_results = []
            for i, screen_id in enumerate(results["ids"][0]):
                formatted_results.append({
                    "id": screen_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Screen memory search failed: {e}")
            return []

    def get_recent_screens(self, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            all_data = self._collection.get()
            if not all_data or not all_data.get("ids"):
                return []
            
            ids = all_data["ids"][-limit:][::-1]
            docs = all_data["documents"][-limit:][::-1]
            metas = all_data["metadatas"][-limit:][::-1]
            
            results = []
            for i, screen_id in enumerate(ids):
                results.append({
                    "id": screen_id,
                    "text": docs[i],
                    "metadata": metas[i]
                })
            return results
        except Exception as e:
            logger.error(f"Failed to get recent screens: {e}")
            return []

    def delete_screen(self, screen_id: str) -> bool:
        try:
            self._collection.delete(ids=[screen_id])
            logger.info(f"Deleted screen memory: {screen_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete screen memory: {e}")
            return False

    def clear_all(self) -> bool:
        try:
            self._client.delete_collection("screen_memory")
            self._collection = self._client.get_or_create_collection(
                name="screen_memory",
                metadata={"description": "OCR text from captured screens"}
            )
            logger.info("Cleared all screen memories")
            return True
        except Exception as e:
            logger.error(f"Failed to clear screen memory: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        try:
            count = self._collection.count()
            return {
                "total_screens": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


_screen_memory_db: Optional[ScreenMemoryDB] = None


def get_screen_memory_db() -> ScreenMemoryDB:
    global _screen_memory_db
    if _screen_memory_db is None:
        _screen_memory_db = ScreenMemoryDB()
    return _screen_memory_db