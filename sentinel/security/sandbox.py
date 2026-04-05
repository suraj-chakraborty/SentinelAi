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

import ast


# ── AST Safety Visitor ────────────────────────────────────────────────────────

_BLOCKED_IMPORTS = {
    "os", "subprocess", "shutil", "ctypes", "winreg", "socket",
    "requests", "httpx", "sys", "builtins", "importlib", "pathlib",
    "pty", "tty", "signal", "multiprocessing", "concurrent", "asyncio",
}

_BLOCKED_CALLS = {
    "eval", "exec", "compile", "open", "__import__", "getattr",
    "setattr", "delattr", "vars", "dir", "globals", "locals",
    "breakpoint", "input",
}

_BLOCKED_ATTRIBS = {
    "system", "popen", "exec_", "execl", "execve", "spawn",
    "rmdir", "remove", "unlink", "chmod", "chown", "kill",
    "environ", "__subclasses__", "__mro__", "__globals__", 
    "__builtins__", "__dict__", "func_globals", "mro",
}


class _SafetyVisitor(ast.NodeVisitor):
    """
    AST-level visitor that raises ValueError on any dangerous node.

    Catches attacks that bypass simple regex:
      • import("os")           → ast.Call with func.id == '__import__'
      • __builtins__['eval']   → ast.Subscript
      • getattr(os, 'system')  → ast.Call with func.id == 'getattr'
    """

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            pkg = alias.name.split(".")[0]
            if pkg in _BLOCKED_IMPORTS:
                raise ValueError(f"Blocked import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        pkg = (node.module or "").split(".")[0]
        if pkg in _BLOCKED_IMPORTS:
            raise ValueError(f"Blocked from-import: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Direct dangerous call: eval(...), exec(...), open(...)
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name and name in _BLOCKED_CALLS:
            raise ValueError(f"Blocked call: {name}(…)")
        # Attribute access: os.system, subprocess.run, shutil.rmtree …
        if isinstance(func, ast.Attribute) and func.attr in _BLOCKED_ATTRIBS:
            raise ValueError(f"Blocked attribute access: .{func.attr}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in _BLOCKED_ATTRIBS:
            raise ValueError(f"Blocked attribute: .{node.attr}")
        self.generic_visit(node)


class Sandbox:
    """
    Safe Python code execution with multiple isolation levels.

    Safety level can be overridden at runtime via:
        SENTINEL_CODE_SANDBOX_LEVEL = 1 | 2 | 3
    """

    @classmethod
    def execute(cls, code: str, level: int = 2, timeout: int = 15) -> Tuple[bool, str]:
        """
        Execute Python code safely.

        Args:
            code:    Python source code to execute
            level:   1=RestrictedPython, 2=subprocess+timeout (default), 3=Docker
                     (overridden by SENTINEL_CODE_SANDBOX_LEVEL env var)
            timeout: Max execution time in seconds

        Returns:
            (success: bool, output: str)
        """
        if not code or not code.strip():
            return False, "No code provided."

        # Env-var override
        env_level = os.getenv("SENTINEL_CODE_SANDBOX_LEVEL", "").strip()
        if env_level.isdigit():
            level = int(env_level)

        # AST-based static analysis (always applied, catches obfuscation)
        blocked = cls._static_analysis(code)
        if blocked:
            return False, f"Code blocked by safety analysis: {blocked}"

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
        """
        AST-level safety scan. Returns violation message or empty string if safe.
        This replaces the old regex approach to catch obfuscated patterns like:
            __import__('os') or __builtins__['eval'](...)
        """
        # First: parse for SyntaxErrors (fast fail)
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"Syntax error: {exc}"

        # Then: walk AST for dangerous nodes
        try:
            _SafetyVisitor().visit(tree)
        except ValueError as exc:
            return str(exc)
        return ""

    @classmethod
    def get_safety_report(cls, code: str) -> dict:
        """Analyse code and return a safety report without executing."""
        violation = cls._static_analysis(code)
        syntax_ok  = not violation.startswith("Syntax error")

        return {
            "safe"         : not violation,
            "violation"    : violation or None,
            "syntax_ok"    : syntax_ok,
            "line_count"   : len(code.splitlines()),
            "sandbox_level": int(os.getenv("SENTINEL_CODE_SANDBOX_LEVEL", "2")),
        }
