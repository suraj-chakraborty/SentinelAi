"""
sentinel/commands/code_whisperer_command.py
────────────────────────────────────────────
Voice Command Plugin for Code Whisperer / Voice Pair-Programmer.

Handles: "start pair programming", "explain this code", "generate a function",
         "debug this error", "rubber duck", "review this code", "refactor this".
"""

import logging
import re
from typing import Optional

from sentinel.modules.code_whisperer import CodeWhispererEngine, CodeLanguage

logger = logging.getLogger("CodeWhispererCommand")


class CodeWhispererCommand:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        llm_fn = getattr(orchestrator, "llm_callback", None) if orchestrator else None
        self.engine = CodeWhispererEngine(llm_callback=llm_fn)

    def execute(self, command: str, entity: Optional[str] = None) -> str:
        cmd = command.lower().strip()

        # 1. End session (checked first to avoid substring collision)
        if any(k in cmd for k in ("end pair", "stop pair", "code whisperer off",
                                   "end coding session", "exit pair")):
            result = self.engine.end_session()
            return (
                f"Pair-programming session ended.\n"
                f"Suggestions given: {result['suggestions_given']} | "
                f"Errors caught: {result['errors_caught']} | "
                f"Debug sessions: {result['debug_sessions']}"
            )

        # 2. Start pair programming session
        if any(k in cmd for k in ("start pair", "pair programming", "code whisperer on",
                                   "start coding session", "be my pair programmer")):
            result = self.engine.start_session()
            return (
                f"👨‍💻 Pair-programming session started!\n"
                f"Language: {result['language']} | Branch: {result.get('branch', 'N/A')}\n"
                f"I'm ready to explain, generate, debug, and review code with you."
            )

        # 3. Explain code
        if any(k in cmd for k in ("explain this", "explain code", "what does this code",
                                   "explain the code", "walk me through")):
            # Extract code from entity or use context
            code = entity or "# No code provided"
            suggestion = self.engine.explain_code(code)
            return f"📖 Code Explanation:\n{suggestion.explanation}"

        # 4. Generate code
        if any(k in cmd for k in ("generate a function", "generate a class",
                                   "create a function", "write a function",
                                   "generate code for", "create code for")):
            # Parse construct type and name
            construct = "function"
            if "class" in cmd:
                construct = "class"
            elif "test" in cmd:
                construct = "test"

            # Try to extract name
            name_match = re.search(r'(?:called|named)\s+(\w+)', cmd)
            name = name_match.group(1) if name_match else "my_function"

            # Description is the full command
            suggestion = self.engine.generate_code(
                description=cmd, construct=construct, name=name
            )
            return (
                f"🔧 Generated {construct} '{name}':\n"
                f"```{suggestion.language.value}\n{suggestion.suggested_code}```\n"
                f"{suggestion.explanation}"
            )

        # 5. Debug / analyze error
        if any(k in cmd for k in ("debug this", "what is this error", "analyze error",
                                   "fix this error", "help with error")):
            error_text = entity or cmd
            result = self.engine.analyze_error(error_text)
            guidance = "\n".join(result["guidance"])
            return f"🐛 Error Analysis ({result['language']}):\n{guidance}"

        # 6. Rubber duck debugging
        if any(k in cmd for k in ("rubber duck", "think out loud", "help me think",
                                   "debug with me", "talk through")):
            thought = entity or cmd
            response = self.engine.rubber_duck(thought)
            return f"🦆 {response}"

        # 7. Refactor suggestions
        if any(k in cmd for k in ("refactor this", "improve this code", "review this code",
                                   "code review", "suggest improvements")):
            code = entity or "# No code provided"
            suggestion = self.engine.suggest_refactor(code)
            return f"🔄 Refactoring Suggestions:\n{suggestion.explanation}"

        # 8. Status
        if any(k in cmd for k in ("code whisperer status", "pair status",
                                   "coding session status")):
            status = self.engine.get_status()
            return (
                f"Code Whisperer: {'Active' if status['session_active'] else 'Inactive'}\n"
                f"File: {status['file'] or 'N/A'} | Language: {status['language']}\n"
                f"Suggestions: {status['suggestions_given']} | "
                f"Errors caught: {status['errors_caught']}"
            )

        return "Code Whisperer command not recognized. Try: start pair programming, explain this code, generate a function, rubber duck."
