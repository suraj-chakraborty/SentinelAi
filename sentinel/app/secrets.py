"""
sentinel/app/secrets.py
─────────────────────────
Secure vault for API keys and master password management.
"""

import os
import sqlite3
import hashlib
import hmac
import struct
import logging
from typing import Optional, Tuple

from sentinel.app.config import SECRETS_DB, ensure_appdata_dir

logger = logging.getLogger("SentinelSecrets")

# ── Hashing Constants ────────────────────────────────────────────────────────

MASTER_META_KEY = "master_hash"
MASTER_META_SALT = "master_salt"

# ── DB Management ────────────────────────────────────────────────────────────

def _ensure_secrets_db():
    """Ensure the secrets SQLite database and tables exist."""
    ensure_appdata_dir()
    db_dir = os.path.dirname(SECRETS_DB)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        
    conn = sqlite3.connect(SECRETS_DB)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB)")
    cur.execute("CREATE TABLE IF NOT EXISTS secrets (name TEXT PRIMARY KEY, salt BLOB, nonce BLOB, ciphertext BLOB)")
    conn.commit()
    return conn

# ── Cryptography Core ────────────────────────────────────────────────────────

def _pbkdf(master: str, salt: bytes, length=32, rounds=200000) -> bytes:
    """Derive a key from a master password using PBKDF2-SHA256."""
    return hashlib.pbkdf2_hmac('sha256', master.encode('utf-8'), salt, rounds, dklen=length)

def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    """Generate a simple SHA256-based keystream for XOR encryption."""
    out = bytearray()
    counter = 0
    while len(out) < length:
        counter_bytes = struct.pack('<Q', counter)
        block = hashlib.sha256(key + nonce + counter_bytes).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])

# AES-GCM Upgrade (optional but recommended)
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:
    AESGCM = None

def encrypt_data(master: str, plaintext: str) -> Tuple[bytes, bytes, bytes]:
    """Encrypt a string using the master password."""
    plaintext_bytes = plaintext.encode('utf-8')
    salt = os.urandom(16)
    key = _pbkdf(master, salt, length=32)
    
    if AESGCM:
        aes = AESGCM(key)
        nonce = os.urandom(12)
        ct = aes.encrypt(nonce, plaintext_bytes, None)
    else:
        raise ImportError("cryptography package required for Phase 3 vaulting.")
        
    return salt, nonce, ct

def decrypt_data(master: str, salt: bytes, nonce: bytes, ciphertext: bytes) -> Optional[str]:
    """Decrypt binary data using the master password."""
    key = _pbkdf(master, salt, length=32)
    
    try:
        if AESGCM:
            aes = AESGCM(key)
            pt_bytes = aes.decrypt(nonce, ciphertext, None)
        else:
            raise ImportError("cryptography package required for Phase 3 vaulting.")
        return pt_bytes.decode('utf-8')
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        return None

def rotate_master_password(old_master: str, new_master: str) -> bool:
    """Re-encrypts all secrets in the database with a new master password."""
    if not verify_master(old_master):
        logger.error("Master password rotation failed: Invalid current password.")
        return False
    
    conn = _ensure_secrets_db()
    cur = conn.cursor()
    cur.execute("SELECT name, salt, nonce, ciphertext FROM secrets")
    rows = cur.fetchall()
    
    try:
        decrypted_secrets = []
        for name, salt, nonce, ct in rows:
            pt = decrypt_data(old_master, salt, nonce, ct)
            if pt is None:
                raise ValueError(f"Could not decrypt secret '{name}' during migration.")
            decrypted_secrets.append((name, pt))
            
        # 1. Update master hash
        set_master_password(new_master)
        
        # 2. Re-encrypt everything
        for name, pt in decrypted_secrets:
            store_secret(name, pt, new_master)
            
        logger.info("Master password rotation successful. All secrets migrated.")
        return True
    except Exception as e:
        logger.error(f"Master password rotation CRITICAL FAILURE: {e}")
        # Note: If this fails mid-way, the old master_hash is still in the DB 
        # unless set_master_password(new_master) already committed.
        return False
    finally:
        conn.close()

# ── Master Password Logic ────────────────────────────────────────────────────

def check_master_exists() -> bool:
    """Check if a master password has been set."""
    conn = _ensure_secrets_db()
    cur = conn.cursor()
    cur.execute("SELECT value FROM meta WHERE key=?", (MASTER_META_KEY,))
    row = cur.fetchone()
    conn.close()
    return row is not None

def set_master_password(master: str):
    """Set the initial master password."""
    conn = _ensure_secrets_db()
    salt = os.urandom(16)
    hashed = _pbkdf(master, salt)
    cur = conn.cursor()
    cur.execute("REPLACE INTO meta(key,value) VALUES(?,?)", (MASTER_META_SALT, salt))
    cur.execute("REPLACE INTO meta(key,value) VALUES(?,?)", (MASTER_META_KEY, hashed))
    conn.commit()
    conn.close()

def verify_master(master: str) -> bool:
    """Verify a master password attempt."""
    if not master: return False
    conn = _ensure_secrets_db()
    cur = conn.cursor()
    cur.execute("SELECT value FROM meta WHERE key=?", (MASTER_META_SALT,))
    row_salt = cur.fetchone()
    cur.execute("SELECT value FROM meta WHERE key=?", (MASTER_META_KEY,))
    row_hash = cur.fetchone()
    conn.close()
    
    if not row_salt or not row_hash:
        # First time — auto-set
        set_master_password(master)
        return True
        
    salt, expected = row_salt[0], row_hash[0]
    attempt = _pbkdf(master, salt)
    return hmac.compare_digest(attempt, expected)

# ── High-Level Vault API ─────────────────────────────────────────────────────

def store_secret(name: str, value: str, master: str) -> bool:
    """Encrypt and store a secret in the vault."""
    if not verify_master(master):
        logger.warning(f"Failed secret store for '{name}': Invalid master.")
        return False
        
    salt, nonce, ct = encrypt_data(master, value)
    conn = _ensure_secrets_db()
    cur = conn.cursor()
    cur.execute("REPLACE INTO secrets(name,salt,nonce,ciphertext) VALUES(?,?,?,?)",
               (name, salt, nonce, ct))
    conn.commit()
    conn.close()
    return True

def fetch_secret(name: str, master: str) -> Optional[str]:
    """Fetch and decrypt a secret from the vault."""
    if not verify_master(master):
        logger.warning(f"Failed secret fetch for '{name}': Invalid master.")
        return None
        
    conn = _ensure_secrets_db()
    cur = conn.cursor()
    cur.execute("SELECT salt, nonce, ciphertext FROM secrets WHERE name=?", (name,))
    row = cur.fetchone()
    conn.close()
    
    if not row: return None
    salt, nonce, ct = row
    return decrypt_data(master, salt, nonce, ct)
