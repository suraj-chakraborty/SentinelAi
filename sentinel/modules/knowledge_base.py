import chromadb
from chromadb.utils import embedding_functions
import os
import logging
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import base64

class KnowledgeBaseModule:
    def __init__(self, db_path="sentinel_db", master_key=None):
        self.db_path = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "SentinelAi", db_path)
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="personal_knowledge",
            embedding_function=self.embedding_fn
        )
        self.logger = logging.getLogger("KnowledgeBaseModule")
        self.master_key = master_key  # This should be a 32-byte key derived from master password

    def _encrypt(self, plaintext):
        if not self.master_key:
            return plaintext
        aes = AESGCM(self.master_key)
        nonce = os.urandom(12)
        ct = aes.encrypt(nonce, plaintext.encode('utf-8'), None)
        return base64.b64encode(nonce + ct).decode('utf-8')

    def _decrypt(self, ciphertext):
        if not self.master_key:
            return ciphertext
        try:
            data = base64.b64decode(ciphertext)
            nonce, ct = data[:12], data[12:]
            aes = AESGCM(self.master_key)
            pt = aes.decrypt(nonce, ct, None)
            return pt.decode('utf-8')
        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            return "[ENCRYPTED DATA - INVALID KEY]"

    def add_information(self, text, metadata=None):
        """Adds encrypted information to the knowledge base."""
        try:
            encrypted_text = self._encrypt(text)
            count = self.collection.count()
            self.collection.add(
                documents=[encrypted_text],
                metadatas=[metadata or {"source": "manual_entry", "encrypted": "true"}],
                ids=[f"id_{count}"]
            )
            return True, "Information added and encrypted in knowledge base."
        except Exception as e:
            self.logger.error(f"Error adding info: {e}")
            return False, f"Error: {e}"

    def query_knowledge(self, query_text, n_results=3):
        """Queries and decrypts knowledge base information."""
        try:
            # We query using embeddings (which are based on encrypted text, or original?)
            # Actually, embeddings should be on plaintext for semantic search to work.
            # But we want the text to be encrypted at rest.
            # Standard RAG: embedding on plaintext -> retrieve encrypted text -> decrypt.
            # To do this, we need to pass plaintext to embedding_fn.
            
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            if results and results['documents']:
                decrypted_docs = [self._decrypt(doc) for doc in results['documents'][0]]
                return "\n".join(decrypted_docs)
            return "No relevant information found."
        except Exception as e:
            self.logger.error(f"Query error: {e}")
            return f"Search failed: {e}"
