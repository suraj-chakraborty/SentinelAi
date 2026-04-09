class JarvisPersona:
    """Simple Jarvis-style persona manager for Phase 1 MVP.

    Exposes a few tone presets and a method to render a system prompt snippet
    that constrains the LLM responses to the chosen persona.
    """

    def __init__(self, tone: str = "calm"):
        self.tone = tone

    def set_tone(self, tone: str) -> None:
        self.tone = tone

    def get_prompt_schip(self) -> str:
        # A compact descriptor used in prompts to steer tone
        return f"You are Jarvis, a local assistant with a {self.tone} and concise speaking style."
