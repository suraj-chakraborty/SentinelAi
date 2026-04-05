import logging
from sentinel.voice.tts import speak
from googlesearch import search

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
            results = []
            # Use googlesearch-python (already installed in venv)
            for url in search(query, num_results=3):
                results.append(url)
            
            if not results:
                return f"I couldn't find any results for '{query}'."
            
            # Briefly describe results
            response = f"I found some results for '{query}'. The top link is {results[0]}."
            # Optionally use AI to summarize if possible
            if self.orchestrator and hasattr(self.orchestrator, "_safe_llm_call"):
                summary = self.orchestrator._safe_llm_call(f"The user searched for '{query}'. Found results: {results}. Summarize briefly.")
                if summary:
                    return summary

            return response
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"I'm sorry, I encountered an error while searching for '{query}': {e}"
