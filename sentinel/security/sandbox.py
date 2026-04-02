"""
sentinel/security/sandbox.py
─────────────────────────────
Sandboxed code execution — Tier 4 Safety Upgrade.

Replaces the bare `subprocess.run` in CodingModule with a
safer execution environment using RestrictedPython, with a
Docker fallback for full isolation.

Safety levels:
    Level 1 (RestrictedPython): Blocks file/network/os access at AST level
    Level 2 (subprocess timeout): 30s timeout, captures all output
    Level 3 (Docker): Full container isolation (requires Docker Desktop)

Usage:
    from sentinel.security.sandbox import Sandbox
    result = Sandbox.execute(code, level=1)
"""

import os
import sys
import re
import subprocess
import tempfile
import logging
import threading
from typing import Tuple

logger = logging.getLogger("SentinelSandbox")


class Sandbox:
    """
    Safe Python code execution with multiple isolation levels.
    """

    # Dangerous patterns that are always blocked
    BLOCKED_PATTERNS = [
        r'\bos\.system\b',
        r'\bsubprocess\b',
        r'\bimport\s+os\b',
        r'\bimport\s+subprocess\b',
        r'\b__import__\b',
        r'\beval\s*\(',
        r'\bexec\s*\(',
        r'\bopen\s*\(',
        r'\brmdir\b',
        r'\bshutil\b',
        r'\bsocket\b',
        r'\brequests\b',
        r'\bhttpx\b',
        r'\bimport\s+ctypes\b',
        r'\bimport\s+winreg\b',
    ]

    @classmethod
    def execute(cls, code: str, level: int = 2, timeout: int = 15) -> Tuple[bool, str]:
        """
        Execute Python code safely.

        Args:
            code:    Python source code to execute
            level:   1=RestrictedPython, 2=subprocess+timeout (default), 3=Docker
            timeout: Max execution time in seconds

        Returns:
            (success: bool, output: str)
        """
        if not code or not code.strip():
            return False, "No code provided."

        # Static analysis blocklist (always applied)
        blocked = cls._static_analysis(code)
        if blocked:
            return False, f"Blocked: {blocked}"

        if level == 1:
            return cls._execute_restricted(code)
        elif level == 3:
            return cls._execute_docker(code, timeout)
        else:
            return cls._execute_subprocess(code, timeout)

    # ─── Level 1: RestrictedPython ─────────────────────────────────────────

    @classmethod
    def _execute_restricted(cls, code: str) -> Tuple[bool, str]:
        """Execute using RestrictedPython (AST-level restrictions)."""
        try:
            from RestrictedPython import compile_restricted, safe_globals, limited_builtins
            from RestrictedPython.Guards import safe_iter_unpack_sequence

            byte_code = compile_restricted(code, filename="<sandbox>", mode="exec")

            restricted_globals = {
                **safe_globals,
                "__builtins__": {
                    **limited_builtins,
                    "print": print,
                    "len": len,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "abs": abs,
                    "max": max,
                    "min": min,
                    "sum": sum,
                    "sorted": sorted,
                    "reversed": reversed,
                    "round": round,
                },
                "_iter_unpack_sequence_": safe_iter_unpack_sequence,
            }

            import io
            from contextlib import redirect_stdout
            output_buffer = io.StringIO()

            with redirect_stdout(output_buffer):
                exec(byte_code, restricted_globals)

            return True, output_buffer.getvalue() or "Execution successful (no output)."

        except ImportError:
            logger.warning("RestrictedPython not installed, falling back to subprocess")
            return cls._execute_subprocess(code, timeout=15)
        except Exception as e:
            return False, f"RestrictedPython error: {e}"

    # ─── Level 2: subprocess + timeout ────────────────────────────────────────

    @classmethod
    def _execute_subprocess(cls, code: str, timeout: int = 15) -> Tuple[bool, str]:
        """Execute in a subprocess with timeout. Safe for most use cases."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".py", delete=False, mode="w", encoding="utf-8"
            ) as f:
                f.write(code)
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                # Restrict environment
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                    "TEMP": os.environ.get("TEMP", ""),
                    "TMP": os.environ.get("TMP", ""),
                }
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                return True, output or "Execution successful (no output)."
            else:
                error = result.stderr.strip()
                # Clean up traceback for readability
                error_lines = [l for l in error.splitlines() if tmp_path not in l]
                return False, "\n".join(error_lines) if error_lines else error

        except subprocess.TimeoutExpired:
            return False, f"Execution timed out after {timeout}s."
        except Exception as e:
            return False, f"Execution error: {e}"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # ─── Level 3: Docker container ─────────────────────────────────────────────

    @classmethod
    def _execute_docker(cls, code: str, timeout: int = 30) -> Tuple[bool, str]:
        """Execute in an isolated Docker container. Requires Docker Desktop."""
        try:
            import docker
            client = docker.from_env()
            output = client.containers.run(
                "python:3.11-slim",
                command=["python", "-c", code],
                mem_limit="128m",
                network_disabled=True,
                remove=True,
                timeout=timeout,
                stderr=True
            )
            return True, output.decode("utf-8").strip() or "Done."
        except ImportError:
            logger.warning("docker SDK not installed. Falling back to subprocess.")
            return cls._execute_subprocess(code, timeout)
        except Exception as e:
            return False, f"Docker execution error: {e}"

    # ─── Static analysis ───────────────────────────────────────────────────────

    @classmethod
    def _static_analysis(cls, code: str) -> str:
        """Scan code for blocked patterns. Returns the violation or empty string."""
        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, code):
                return f"Forbidden pattern: {pattern}"
        return ""

    @classmethod
    def get_safety_report(cls, code: str) -> dict:
        """Analyze code and return a safety report without executing."""
        violations = []
        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, code):
                violations.append(pattern)

        import ast
        syntax_ok = True
        syntax_error = None
        try:
            ast.parse(code)
        except SyntaxError as e:
            syntax_ok = False
            syntax_error = str(e)

        return {
            "safe": len(violations) == 0 and syntax_ok,
            "violations": violations,
            "syntax_ok": syntax_ok,
            "syntax_error": syntax_error,
            "line_count": len(code.splitlines()),
        }
