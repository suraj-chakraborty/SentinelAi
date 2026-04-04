import subprocess
import os
import tempfile
import ast
import logging

class CodingModule:
    def __init__(self):
        self.logger = logging.getLogger("CodingModule")

    def validate_code(self, code):
        """Basic syntax check using AST."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, str(e)

    def execute_python_code(self, code, retry_count=0, max_retries=2, llm_callback=None):
        """Executes Python code safely and attempts self-correction if it fails."""
        is_valid, error = self.validate_code(code)
        if not is_valid:
            if retry_count < max_retries and llm_callback:
                self.logger.info(f"Syntax error, attempting self-correction (retry {retry_count+1})...")
                correction_prompt = f"The following Python code has a syntax error: {error}\n\nCode:\n```python\n{code}\n```\nPlease fix the code and return only the corrected code block."
                corrected_code_response = llm_callback(correction_prompt)
                corrected_code = self._extract_code(corrected_code_response)
                return self.execute_python_code(corrected_code, retry_count + 1, max_retries, llm_callback)
            return f"Syntax error: {error}"

        try:
            try:
                from sentinel.security.sandbox import Sandbox

                ok, output = Sandbox.execute(code, level=2, timeout=30)
                if ok:
                    return output or "Execution successful (no output)."
                if retry_count < max_retries and llm_callback:
                    self.logger.info(
                        "Sandbox execution failed, attempting self-correction (retry %s)...",
                        retry_count + 1,
                    )
                    correction_prompt = (
                        f"The following Python code failed:\n{output}\n\n"
                        f"Original Code:\n```python\n{code}\n```\n"
                        "Please fix the code and return only the corrected code block."
                    )
                    corrected_code_response = llm_callback(correction_prompt)
                    corrected_code = self._extract_code(corrected_code_response)
                    return self.execute_python_code(
                        corrected_code, retry_count + 1, max_retries, llm_callback
                    )
                return f"Execution error: {output}"
            except ImportError:
                pass

            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
                tmp.write(code.encode("utf-8"))
                tmp_path = tmp.name

            try:
                result = subprocess.run(
                    ["python", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    return result.stdout or "Execution successful (no output)."
                stderr = result.stderr
                if retry_count < max_retries and llm_callback:
                    self.logger.info(
                        "Runtime error, attempting self-correction (retry %s)...",
                        retry_count + 1,
                    )
                    correction_prompt = (
                        f"The following Python code failed with a runtime error:\n{stderr}\n\n"
                        f"Original Code:\n```python\n{code}\n```\n"
                        "Please analyze the error, fix the code, and return only the corrected code block."
                    )
                    corrected_code_response = llm_callback(correction_prompt)
                    corrected_code = self._extract_code(corrected_code_response)
                    return self.execute_python_code(
                        corrected_code, retry_count + 1, max_retries, llm_callback
                    )
                return f"Execution error: {stderr}"
            except subprocess.TimeoutExpired:
                return "Execution timed out (max 30s)."
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception as e:
            self.logger.error(f"Error executing code: {e}")
            return f"Error executing code: {e}"

    def _extract_code(self, response):
        """Helper to extract code from LLM response blocks."""
        if "```python" in response:
            return response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            return response.split("```")[1].split("```")[0].strip()
        return response.strip()

    def generate_and_execute(self, prompt, llm_callback):
        """Generates code using an LLM and executes it with self-correction."""
        code_prompt = f"Write a Python script for the following task: {prompt}. Return only the code in a code block."
        generated_code_response = llm_callback(code_prompt)
        code = self._extract_code(generated_code_response)

        if code:
            return self.execute_python_code(code, llm_callback=llm_callback)
        return "No code generated."
