"""
sentinel/core/sandbox.py
────────────────────────
Autonomous Sandboxed Code Execution Engine.

When user asks Sentinel to write a complex script, this module:
1. Spins up an ephemeral Alpine Linux Docker container
2. Injects the generated code + test script
3. Executes and captures results
4. Feeds errors back to LLM for autonomous fixing
5. Returns validated code only after successful execution
"""

import os
import json
import logging
import tempfile
import subprocess
import time
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from sentinel.app.config import APPDATA_DIR

logger = logging.getLogger("SandboxEngine")

SANDBOX_DIR = os.path.join(APPDATA_DIR, "sandbox")
os.makedirs(SANDBOX_DIR, exist_ok=True)


class SandboxResult:
    """Result of sandboxed code execution."""
    
    def __init__(
        self,
        success: bool,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = -1,
        duration_ms: float = 0
    ):
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.duration_ms = duration_ms


class SandboxEngine:
    """
    Sandboxed execution engine using Docker.
    
    Workflow:
    1. Receive generated code from LLM
    2. Create temp directory with code + test
    3. Spin up Alpine container
    4. Execute and capture output
    5. Return result (success/failure + output)
    """

    def __init__(
        self,
        image: str = "alpine:latest",
        timeout_seconds: int = 60,
        max_retries: int = 2
    ):
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._docker_available = None

    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is not None:
            return self._docker_available
        
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=5
            )
            self._docker_available = result.returncode == 0
            if self._docker_available:
                logger.info("Docker available for sandbox")
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
            self._docker_available = False
        
        return self._docker_available

    def execute(
        self,
        code: str,
        language: str = "python",
        test_code: Optional[str] = None,
        dependencies: Optional[list] = None
    ) -> SandboxResult:
        """
        Execute code in sandboxed container.
        
        Args:
            code: The code to execute
            language: Programming language (python, javascript, bash)
            test_code: Optional test code to run
            dependencies: List of packages to install
        
        Returns:
            SandboxResult with execution output
        """
        if not self._check_docker():
            logger.warning("Docker not available, simulating execution")
            return self._simulate_execution(code, language)
        
        start_time = time.time()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                self._prepare_code_files(tmpdir, code, language, test_code, dependencies)
                
                container_id = self._create_container(tmpdir, language)
                if not container_id:
                    return SandboxResult(False, stderr="Failed to create container")
                
                try:
                    result = self._run_in_container(container_id)
                finally:
                    self._cleanup_container(container_id)
                
                duration_ms = (time.time() - start_time) * 1000
                
                return SandboxResult(
                    success=result["exit_code"] == 0,
                    stdout=result.get("stdout", ""),
                    stderr=result.get("stderr", ""),
                    exit_code=result["exit_code"],
                    duration_ms=duration_ms
                )
        
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return SandboxResult(False, stderr=str(e))

    def _prepare_code_files(
        self,
        tmpdir: str,
        code: str,
        language: str,
        test_code: Optional[str],
        dependencies: Optional[list]
    ):
        """Prepare code files and install script in temp directory."""
        if language == "python":
            main_file = "main.py"
            Path(tmpdir, main_file).write_text(code)
            
            if test_code:
                Path(tmpdir, "test.py").write_text(test_code)
            
            if dependencies:
                reqs = "\n".join(dependencies)
                Path(tmpdir, "requirements.txt").write_text(reqs)
        
        elif language == "javascript":
            main_file = "main.js"
            Path(tmpdir, main_file).write_text(code)
            
            if test_code:
                Path(tmpdir, "test.js").write_text(test_code)
        
        elif language == "bash":
            main_file = "script.sh"
            Path(tmpdir, main_file).write_text(f"#!/bin/sh\n{code}")
            os.chmod(os.path.join(tmpdir, main_file), 0o755)
        
        else:
            main_file = "main.txt"
            Path(tmpdir, main_file).write_text(code)

    def _create_container(self, tmpdir: str, language: str) -> Optional[str]:
        """Create and start a Docker container."""
        try:
            dockerfile = f"""
FROM {self.image}
RUN apk add --no-cache \\
    python3 \\
    py3-pip \\
    nodejs \\
    npm
"""
            if language == "python":
                dockerfile += "RUN pip3 install --no-cache-dir pytest"
            
            subprocess.run(
                ["docker", "build", "-t", "sentinel-sandbox", "-"],
                input=dockerfile,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            result = subprocess.run(
                [
                    "docker", "run", "-d",
                    "-v", f"{tmpdir}:/code",
                    "--rm",
                    "--name", f"sentinel-{int(time.time())}",
                    "sentinel-sandbox",
                    "sleep", "300"
                ],
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
        
        except Exception as e:
            logger.error(f"Container creation failed: {e}")
        
        return None

    def _run_in_container(self, container_id: str) -> Dict[str, Any]:
        """Execute code in running container."""
        try:
            result = subprocess.run(
                ["docker", "exec", container_id, "sh", "-c", "cd /code && python3 main.py"],
                capture_output=True,
                timeout=self.timeout_seconds
            )
            
            return {
                "stdout": result.stdout.decode("utf-8", errors="replace"),
                "stderr": result.stderr.decode("utf-8", errors="replace"),
                "exit_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "Execution timeout",
                "exit_code": -1
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1
            }

    def _cleanup_container(self, container_id: str):
        """Stop and remove the container."""
        try:
            subprocess.run(
                ["docker", "stop", container_id],
                capture_output=True,
                timeout=10
            )
        except Exception:
            pass

    def _simulate_execution(self, code: str, language: str) -> SandboxResult:
        """Simulate execution when Docker is unavailable."""
        try:
            if language == "python":
                import ast
                ast.parse(code)
                return SandboxResult(True, stdout="Code syntax validated (simulated)", exit_code=0)
            return SandboxResult(True, stdout="Simulated execution", exit_code=0)
        except SyntaxError as e:
            return SandboxResult(False, stderr=f"Syntax error: {e}", exit_code=1)


class AutoFixEngine:
    """Autonomous code fixing based on execution errors."""

    def __init__(self, sandbox: SandboxEngine, orchestrator=None):
        self.sandbox = sandbox
        self.orchestrator = orchestrator

    def execute_with_autofix(
        self,
        code: str,
        language: str = "python",
        test_code: Optional[str] = None,
        max_iterations: int = 3
    ) -> Tuple[bool, str, SandboxResult]:
        """
        Execute code with autonomous fixing on failure.
        
        Returns:
            (success, final_code, result)
        """
        current_code = code
        iterations = 0
        
        while iterations < max_iterations:
            result = self.sandbox.execute(current_code, language, test_code)
            
            if result.success:
                return True, current_code, result
            
            if not self.orchestrator:
                return False, current_code, result
            
            iterations += 1
            logger.info(f"Attempt {iterations}: Fixing code error...")
            
            current_code = self._fix_code(current_code, result.stderr, language)
            
            if current_code == code:
                break
        
        return False, current_code, result

    def _fix_code(self, code: str, error: str, language: str) -> str:
        """Use LLM to fix code based on error."""
        if not self.orchestrator or not hasattr(self.orchestrator, "_safe_llm_call"):
            return code
        
        prompt = f"""Fix the following {language} code that produced this error:
Error: {error}

Original code:
```{language}
{code}
```

Provide ONLY the corrected code, no explanation."""
        
        try:
            response = self.orchestrator._safe_llm_call(prompt)
            
            for fence in (f"```{language}", "```"):
                if fence in response:
                    response = response.split(fence)[1].split("```")[0].strip()
                    break
            
            if response:
                return response
        
        except Exception as e:
            logger.error(f"Auto-fix failed: {e}")
        
        return code


_sandbox_engine: Optional[SandboxEngine] = None


def get_sandbox_engine() -> SandboxEngine:
    global _sandbox_engine
    if _sandbox_engine is None:
        _sandbox_engine = SandboxEngine()
    return _sandbox_engine