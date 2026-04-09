import logging
try:
    from sentinel.voice.tts import speak
except Exception:
    try:
        from sentinel.app.voice import speak
    except Exception:
        def speak(*args, **kwargs):
            pass
# googlesearch is loaded lazily inside execute to avoid hard dependency in tests

logger = logging.getLogger("SearchWeb")

class SearchWeb:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def execute(self, command: str, entity: str) -> str:
        """Perform a web search for the entity."""
        query = entity or command
        if not query:
            return "What would you like me to search for?"

        speak(f"Searching the web for {query}...", block=False)
        logger.info(f"Searching for: {query}")
        
        try:
            # Lazy import: only import googlesearch if available in this environment
            from googlesearch import search  # type: ignore
            results = [url for url in search(query, num_results=3)]
            if not results:
                return None

            # Briefly describe results
            response = f"I found some results for '{query}'. The top link is {results[0]}."
            # Optionally use AI to summarize if possible
            if self.orchestrator and hasattr(self.orchestrator, "_safe_llm_call"):
                summary = self.orchestrator._safe_llm_call(
                    f"The user searched for '{query}'. Found results: {results}. Summarize briefly."
                )
                if summary:
                    return summary

            return response
        except ImportError:
            # googlesearch not installed; skip web search gracefully
            return None
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return None
