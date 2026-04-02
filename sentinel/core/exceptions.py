"""
sentinel/core/exceptions.py
─────────────────────────────
Typed Exception Hierarchy — Tier 4 Production Hardening.

Provides structured error handling across SentinelAI modules.
"""

class SentinelError(Exception):
    """Base exception for all SentinelAI errors."""
    def __init__(self, message="A Sentinel error occurred", module=None, original_error=None):
        self.message = message
        self.module = module
        self.original_error = original_error
        super().__init__(self.message)

    def __str__(self):
        mod_prefix = f"[{self.module}] " if self.module else ""
        orig_suffix = f" (Cause: {self.original_error})" if self.original_error else ""
        return f"{mod_prefix}{self.message}{orig_suffix}"

class HardwareError(SentinelError):
    """Raised when hardware components (mic, camera) fail."""
    pass

class LLMServiceError(SentinelError):
    """Raised when an online or offline LLM fails."""
    pass

class VaultError(SentinelError):
    """Raised for AES-GCM encryption/decryption failures."""
    pass

class ActionExecutionError(SentinelError):
    """Raised when an agentic action or automation fails."""
    pass

class SandboxError(SentinelError):
    """Raised when sandboxed code execution violates policy or fails."""
    pass
