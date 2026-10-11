"""
sentinel/modules/code_whisperer.py
───────────────────────────────────
Voice Pair-Programmer / Code Whisperer Module.

Real-time voice-driven coding assistant that:
  1. Watches active editor context (file, language, cursor position).
  2. Provides voice-triggered code completions, refactoring suggestions.
  3. Explains code in natural language on demand.
  4. Detects and announces errors/warnings in real-time.
  5. Generates boilerplate (functions, classes, tests) from voice descriptions.
  6. Supports "rubber duck debugging" — talks through logic with the user.
  7. Integrates with Git for contextual code review.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("CodeWhisperer")


class CodeLanguage(Enum):
    PYTHON     = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA       = "java"
    CPP        = "cpp"
    CSHARP     = "csharp"
    RUST       = "rust"
    GO         = "go"
    HTML       = "html"
    CSS        = "css"
    SQL        = "sql"
    SHELL      = "shell"
    UNKNOWN    = "unknown"


class CodeAction(Enum):
    EXPLAIN       = "explain"
    COMPLETE      = "complete"
    REFACTOR      = "refactor"
    DEBUG         = "debug"
    GENERATE      = "generate"
    REVIEW        = "review"
    TEST_GENERATE = "test_generate"
    OPTIMIZE      = "optimize"
    DOCUMENT      = "document"


# File extension to language mapping
_EXT_LANGUAGE_MAP = {
    ".py": CodeLanguage.PYTHON, ".pyw": CodeLanguage.PYTHON,
    ".js": CodeLanguage.JAVASCRIPT, ".jsx": CodeLanguage.JAVASCRIPT,
    ".ts": CodeLanguage.TYPESCRIPT, ".tsx": CodeLanguage.TYPESCRIPT,
    ".java": CodeLanguage.JAVA,
    ".cpp": CodeLanguage.CPP, ".cc": CodeLanguage.CPP, ".cxx": CodeLanguage.CPP,
    ".h": CodeLanguage.CPP, ".hpp": CodeLanguage.CPP,
    ".cs": CodeLanguage.CSHARP,
    ".rs": CodeLanguage.RUST,
    ".go": CodeLanguage.GO,
    ".html": CodeLanguage.HTML, ".htm": CodeLanguage.HTML,
    ".css": CodeLanguage.CSS, ".scss": CodeLanguage.CSS,
    ".sql": CodeLanguage.SQL,
    ".sh": CodeLanguage.SHELL, ".bash": CodeLanguage.SHELL,
    ".bat": CodeLanguage.SHELL, ".ps1": CodeLanguage.SHELL,
}


@dataclass
class EditorContext:
    """Current editor state for context-aware assistance."""
    file_path: Optional[str] = None
    language: CodeLanguage = CodeLanguage.UNKNOWN
    cursor_line: int = 0
    cursor_col: int = 0
    selected_text: str = ""
    visible_code: str = ""
    file_content: str = ""
    project_root: Optional[str] = None
    git_branch: Optional[str] = None
    recent_errors: List[str] = field(default_factory=list)


@dataclass
class CodeSuggestion:
    """A code suggestion or generation result."""
    action: CodeAction
    language: CodeLanguage
    original_code: str
    suggested_code: str
    explanation: str
    confidence: float
    line_range: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EditorWatcher:
    """Watches the active editor for context changes."""

    def __init__(self):
        self._context = EditorContext()

    def detect_active_file(self) -> Optional[str]:
        """Detect the currently active file from VS Code or other editors."""
        try:
            # Try VS Code workspace detection
            from sentinel.utils.window_context import get_active_window_info
            info = get_active_window_info()
            title = info.get("title", "")

            # VS Code title format: "filename — folder - Visual Studio Code"
            if "Visual Studio Code" in title:
                parts = title.split(" — ")
                if parts:
                    filename = parts[0].strip()
                    if "." in filename:
                        return filename
        except Exception:
            pass
        return None

    def update_context(self, file_path: Optional[str] = None,
                       content: str = "", cursor_line: int = 0,
                       selected_text: str = "") -> EditorContext:
        """Update editor context manually or from detection."""
        if file_path:
            self._context.file_path = file_path
            ext = os.path.splitext(file_path)[1].lower()
            self._context.language = _EXT_LANGUAGE_MAP.get(ext, CodeLanguage.UNKNOWN)

        if content:
            self._context.file_content = content
            lines = content.split("\n")
            start = max(0, cursor_line - 20)
            end = min(len(lines), cursor_line + 20)
            self._context.visible_code = "\n".join(lines[start:end])

        self._context.cursor_line = cursor_line
        self._context.selected_text = selected_text

        # Detect git branch
        if file_path:
            project_root = self._find_project_root(os.path.dirname(file_path))
            self._context.project_root = project_root
            if project_root:
                self._context.git_branch = self._get_git_branch(project_root)

        return self._context

    def _find_project_root(self, start_dir: str) -> Optional[str]:
        """Walk up to find .git directory."""
        current = start_dir
        for _ in range(10):
            if os.path.exists(os.path.join(current, ".git")):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        return None

    def _get_git_branch(self, repo_path: str) -> Optional[str]:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=repo_path, capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    @property
    def context(self) -> EditorContext:
        return self._context


class ErrorDetector:
    """Detects and parses code errors from various sources."""

    # Common error patterns
    _PATTERNS = {
        CodeLanguage.PYTHON: [
            re.compile(r'File "(.+)", line (\d+).*\n\s*(.+)\n(\w+Error:.+)', re.MULTILINE),
            re.compile(r'(\w+Error):\s*(.+)'),
            re.compile(r'SyntaxError:\s*(.+)'),
        ],
        CodeLanguage.JAVASCRIPT: [
            re.compile(r'(\w+Error):\s*(.+)'),
            re.compile(r'at\s+(.+)\s+\((.+):(\d+):(\d+)\)'),
        ],
    }

    def detect_errors(self, output: str, language: CodeLanguage) -> List[Dict[str, Any]]:
        """Parse error output and return structured error information."""
        errors = []
        patterns = self._PATTERNS.get(language, [])

        for pattern in patterns:
            for match in pattern.finditer(output):
                errors.append({
                    "type": match.group(1) if match.lastindex >= 1 else "Error",
                    "message": match.group(match.lastindex) if match.lastindex else "",
                    "full_match": match.group(0)[:200],
                    "language": language.value,
                })

        return errors


class CodeGenerator:
    """Generates code boilerplate from voice descriptions."""

    _TEMPLATES = {
        (CodeLanguage.PYTHON, "function"): (
            'def {name}({params}):\n'
            '    """{description}"""\n'
            '    pass\n'
        ),
        (CodeLanguage.PYTHON, "class"): (
            'class {name}:\n'
            '    """{description}"""\n\n'
            '    def __init__(self):\n'
            '        pass\n'
        ),
        (CodeLanguage.PYTHON, "test"): (
            'def test_{name}():\n'
            '    """{description}"""\n'
            '    # Arrange\n\n'
            '    # Act\n\n'
            '    # Assert\n'
            '    assert True\n'
        ),
        (CodeLanguage.JAVASCRIPT, "function"): (
            'function {name}({params}) {{\n'
            '  // {description}\n'
            '}}\n'
        ),
        (CodeLanguage.JAVASCRIPT, "class"): (
            'class {name} {{\n'
            '  constructor() {{\n'
            '    // {description}\n'
            '  }}\n'
            '}}\n'
        ),
        (CodeLanguage.TYPESCRIPT, "function"): (
            'function {name}({params}): void {{\n'
            '  // {description}\n'
            '}}\n'
        ),
    }

    def generate(self, language: CodeLanguage, construct: str,
                 name: str, description: str = "",
                 params: str = "") -> CodeSuggestion:
        """Generate code from template."""
        template = self._TEMPLATES.get((language, construct))
        if not template:
            # Fallback to Python
            template = self._TEMPLATES.get((CodeLanguage.PYTHON, construct),
                                           'def {name}({params}):\n    pass\n')

        code = template.format(
            name=name, params=params,
            description=description or f"TODO: Implement {name}"
        )

        return CodeSuggestion(
            action=CodeAction.GENERATE,
            language=language,
            original_code="",
            suggested_code=code,
            explanation=f"Generated {construct} '{name}' in {language.value}",
            confidence=0.95,
        )


class CodeWhispererEngine:
    """
    Master Voice Pair-Programmer controller.

    Orchestrates editor watching, error detection, code generation,
    and contextual coding assistance.
    """

    def __init__(self, llm_callback: Optional[Callable] = None):
        self._editor = EditorWatcher()
        self._error_detector = ErrorDetector()
        self._code_gen = CodeGenerator()
        self._llm_callback = llm_callback
        self._session_active = False
        self._suggestions_given = 0
        self._errors_caught = 0
        self._debug_sessions = 0
        logger.info("CodeWhispererEngine initialized")

    def start_session(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Start a pair-programming session."""
        self._session_active = True
        ctx = self._editor.update_context(file_path=file_path)
        return {
            "success": True,
            "session_active": True,
            "file": ctx.file_path,
            "language": ctx.language.value,
            "branch": ctx.git_branch,
        }

    def end_session(self) -> Dict[str, Any]:
        """End the pair-programming session."""
        self._session_active = False
        return {
            "success": True,
            "suggestions_given": self._suggestions_given,
            "errors_caught": self._errors_caught,
            "debug_sessions": self._debug_sessions,
        }

    def explain_code(self, code: str, language: Optional[CodeLanguage] = None) -> CodeSuggestion:
        """Generate a natural language explanation of code."""
        lang = language or self._editor.context.language

        # Use LLM if available for deep explanation
        if self._llm_callback:
            try:
                prompt = f"Explain this {lang.value} code concisely:\n```{lang.value}\n{code}\n```"
                explanation = self._llm_callback(prompt)
                self._suggestions_given += 1
                return CodeSuggestion(
                    action=CodeAction.EXPLAIN,
                    language=lang,
                    original_code=code,
                    suggested_code="",
                    explanation=explanation,
                    confidence=0.90,
                )
            except Exception:
                pass

        # Fallback: basic structural analysis
        lines = code.strip().split("\n")
        analysis_parts = [f"This is a {lang.value} code block with {len(lines)} lines."]

        # Detect key constructs
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def "):
                func_name = stripped.split("(")[0].replace("def ", "")
                analysis_parts.append(f"Defines function '{func_name}'.")
            elif stripped.startswith("class "):
                cls_name = stripped.split("(")[0].split(":")[0].replace("class ", "")
                analysis_parts.append(f"Defines class '{cls_name}'.")
            elif stripped.startswith("import ") or stripped.startswith("from "):
                analysis_parts.append(f"Imports: {stripped}")

        self._suggestions_given += 1
        return CodeSuggestion(
            action=CodeAction.EXPLAIN,
            language=lang,
            original_code=code,
            suggested_code="",
            explanation=" ".join(analysis_parts),
            confidence=0.70,
        )

    def generate_code(self, description: str, construct: str = "function",
                      name: str = "my_function",
                      language: Optional[CodeLanguage] = None) -> CodeSuggestion:
        """Generate code from a voice description."""
        lang = language or self._editor.context.language
        if lang == CodeLanguage.UNKNOWN:
            lang = CodeLanguage.PYTHON

        suggestion = self._code_gen.generate(
            language=lang, construct=construct,
            name=name, description=description,
        )
        self._suggestions_given += 1
        return suggestion

    def analyze_error(self, error_output: str,
                      language: Optional[CodeLanguage] = None) -> Dict[str, Any]:
        """Analyze an error output and provide debugging guidance."""
        lang = language or self._editor.context.language
        errors = self._error_detector.detect_errors(error_output, lang)
        self._errors_caught += len(errors)

        guidance = []
        for err in errors:
            guidance.append(f"❌ {err['type']}: {err['message']}")

        if not errors:
            guidance.append("No structured errors detected in the output.")

        self._debug_sessions += 1
        return {
            "errors_found": len(errors),
            "errors": errors,
            "guidance": guidance,
            "language": lang.value,
        }

    def rubber_duck(self, thought: str) -> str:
        """Rubber duck debugging — Sentinel listens and asks clarifying questions."""
        questions = [
            f"I hear you thinking about: '{thought[:100]}'. "
            "Let me ask: What specifically is the expected behavior versus what you're seeing?",
            "Have you checked the input data format at each stage?",
            "Could there be an off-by-one error or a race condition?",
            "What was the last thing you changed before this broke?",
        ]

        import random
        selected = random.choice(questions)

        if self._llm_callback:
            try:
                prompt = (
                    f"The developer is rubber-duck debugging. They said: '{thought}'. "
                    "Ask one insightful clarifying question to help them find the bug."
                )
                selected = self._llm_callback(prompt)
            except Exception:
                pass

        return selected

    def suggest_refactor(self, code: str,
                          language: Optional[CodeLanguage] = None) -> CodeSuggestion:
        """Suggest refactoring improvements for code."""
        lang = language or self._editor.context.language
        suggestions = []

        # Basic static analysis
        lines = code.split("\n")
        if len(lines) > 50:
            suggestions.append("Function is very long (>50 lines). Consider breaking it into smaller functions.")
        if code.count("if ") > 5:
            suggestions.append("High branching complexity. Consider using a strategy pattern or lookup table.")
        if "except:" in code or "except Exception:" in code:
            suggestions.append("Broad exception catching detected. Consider catching specific exceptions.")
        if not suggestions:
            suggestions.append("Code looks clean! No obvious refactoring needed.")

        self._suggestions_given += 1
        return CodeSuggestion(
            action=CodeAction.REFACTOR,
            language=lang,
            original_code=code,
            suggested_code="",
            explanation="\n".join(f"• {s}" for s in suggestions),
            confidence=0.75,
        )

    def get_status(self) -> Dict[str, Any]:
        """Return code whisperer session status."""
        ctx = self._editor.context
        return {
            "session_active": self._session_active,
            "file": ctx.file_path,
            "language": ctx.language.value,
            "branch": ctx.git_branch,
            "suggestions_given": self._suggestions_given,
            "errors_caught": self._errors_caught,
            "debug_sessions": self._debug_sessions,
        }
