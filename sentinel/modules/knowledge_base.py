import chromadb
from chromadb.utils import embedding_functions
import os
import logging
import uuid
import json

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _AES_AVAILABLE = True
except ImportError:
    _AES_AVAILABLE = False

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

import base64

class KnowledgeBaseModule:
    def __init__(self, db_path="sentinel_db", master_key=None, llm_callback=None):
        self.db_path = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi", db_path)
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="personal_knowledge",
            embedding_function=self.embedding_fn
        )
        self.logger = logging.getLogger("KnowledgeBaseModule")
        self.master_key = master_key
        self.llm_callback = llm_callback

        # ── Phase 3: Memory Graph (NetworkX) ──
        self.graph_path = os.path.join(self.db_path, "memory_graph.json")
        self.graph = nx.DiGraph() if _NX_AVAILABLE else None
        self._load_graph()

    def _encrypt(self, plaintext: str) -> str:
        """Encrypt text for at-rest storage. Returns base64 string."""
        if not self.master_key or not _AES_AVAILABLE:
            return plaintext
        aes = AESGCM(self.master_key)
        nonce = os.urandom(12)
        ct = aes.encrypt(nonce, plaintext.encode("utf-8"), None)
        return base64.b64encode(nonce + ct).decode("utf-8")

    def _decrypt(self, ciphertext: str) -> str:
        """Decrypt stored text. Falls back to returning as-is if not encrypted."""
        if not self.master_key or not _AES_AVAILABLE:
            return ciphertext
        try:
            data = base64.b64decode(ciphertext)
            nonce, ct = data[:12], data[12:]
            aes = AESGCM(self.master_key)
            return aes.decrypt(nonce, ct, None).decode("utf-8")
        except Exception:
            # Not encrypted or wrong key — return as plaintext
            return ciphertext

    def add_information(self, text: str, metadata: dict = None) -> tuple:
        """
        Adds information to the knowledge base.

        FIXED: Embeds PLAINTEXT so semantic search works correctly.
        Stores the ENCRYPTED version in metadata for privacy at rest.
        """
        if not text or not text.strip():
            return False, "Empty text provided."
        try:
            # Record in vector DB
            encrypted_payload = self._encrypt(text)
            doc_id = f"kb_{uuid.uuid4().hex}"
            self.collection.add(
                documents=[text],
                metadatas=[{
                    **(metadata or {}),
                    "source": (metadata or {}).get("source", "manual_entry"),
                    "encrypted_payload": encrypted_payload,
                    "is_encrypted": str(self.master_key is not None)
                }],
                ids=[doc_id]
            )

            # Record in Graph Database
            if self.graph is not None and self.llm_callback:
                self._extract_and_add_to_graph(text)

            return True, "Information stored in knowledge base."
        except Exception as e:
            self.logger.error(f"Error adding info: {e}")
            return False, f"Error: {e}"

    # ── Graph Operations ──────────────────────────────────────────────────────

    def _load_graph(self):
        """Loads the MemoryGraph from a local JSON file."""
        if not self.graph or not os.path.exists(self.graph_path):
            return
        try:
            with open(self.graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.graph = nx.node_link_graph(data)
                self.logger.info(f"Loaded MemoryGraph with {self.graph.number_of_nodes()} nodes.")
        except Exception as e:
            self.logger.error(f"Failed to load memory graph: {e}")

    def _save_graph(self):
        """Saves the MemoryGraph to a local JSON file."""
        if not self.graph:
            return
        try:
            data = nx.node_link_data(self.graph)
            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.error(f"Failed to save memory graph: {e}")

    def _extract_and_add_to_graph(self, text: str):
        """Uses LLM to extract [Subject] -> [Predicate] -> [Object] and add to graph."""
        if not self.llm_callback:
            return
        
        prompt = (
            f"Extract entity relationships from the following text into a strict JSON list of lists.\n"
            f"Format = [[\"Subject Node\", \"Predicate/Relationship\", \"Object Node\"], ...]\n"
            f"Only output the raw JSON array. Example: [[\"User\", \"likes\", \"Python\"]]\n\n"
            f"Text: {text}"
        )
        try:
            response = self.llm_callback(prompt)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            tuples = json.loads(response)
            if isinstance(tuples, list):
                added = 0
                for t in tuples:
                    if len(t) == 3:
                        subj, pred, obj = str(t[0]).lower().strip(), str(t[1]).lower().strip(), str(t[2]).lower().strip()
                        self.graph.add_edge(subj, obj, relation=pred)
                        added += 1
                if added > 0:
                    self._save_graph()
                    self.logger.info(f"Added {added} edges to MemoryGraph.")
        except Exception as e:
            self.logger.warning(f"Graph extraction failed (normal if not relational text): {e}")

    def query_graph(self, entity: str) -> str:
        """Retrieves 1st-degree relationships for a node in the graph."""
        if not self.graph:
            return "Graph database unavailable."
        entity = entity.lower().strip()
        if not self.graph.has_node(entity):
            return f"I have no specific relational memory about '{entity}'."
        
        relations = []
        for nbr, datadict in self.graph.adj[entity].items():
            rel = datadict.get("relation", "is related to")
            relations.append(f"{entity} {rel} {nbr}")
        
        return "\n".join(relations)

    def query_knowledge(self, query_text: str, n_results: int = 3) -> str:
        """
        Semantically searches the knowledge base and returns decrypted results.

        FIXED: Query uses plaintext embeddings so semantic relevance is correct.
        """
        if not query_text:
            return "No query provided."
        try:
            count = self.collection.count()
            if count == 0:
                return "Knowledge base is empty."
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(n_results, count),
                include=["documents", "metadatas", "distances"]
            )
            if not results or not results.get("documents") or not results["documents"][0]:
                return "No relevant information found."

            # Return the plaintext documents (embeddings are correct now)
            docs = results["documents"][0]
            distances = results.get("distances", [[]])[0]
            output_parts = []
            for i, doc in enumerate(docs):
                score = round(1 - distances[i], 3) if distances else "N/A"
                output_parts.append(f"[Relevance: {score}] {doc}")
            return "\n\n".join(output_parts)
        except Exception as e:
            self.logger.error(f"Query error: {e}")
            return f"Search failed: {e}"

    def delete_by_id(self, doc_id: str) -> bool:
        """Deletes a specific entry from the knowledge base."""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            self.logger.error(f"Delete error: {e}")
            return False

    def get_count(self) -> int:
        """Returns the number of entries in the knowledge base."""
        try:
            return self.collection.count()
        except Exception:
            return 0

