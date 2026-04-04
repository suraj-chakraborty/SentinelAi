import re

def preprocess_command(text: str) -> str:
    """
    Cleans the input command by:
    1. Lowercasing and stripping whitespace.
    2. Removing common filler words and the assistant's name.
    3. Collapsing multiple spaces.
    """
    if not text:
        return ""
        
    text = text.lower().strip()
    
    # 1. Remove common punctuation
    text = re.sub(r'[.!?,:;]', '', text)

    # 2. Define fillers to remove (keep semantics: do not strip "tell me"/"what is"/"about" — breaks Q&A routing)
    fillers = [
        "please", "can you", "could you", "hey", "hi", "hello",
        "sentinel", "ai", "assistant", "would you mind", "now",
        "show me", "give me",
    ]
    
    # Sort fillers by length (descending) to avoid partial matches
    fillers.sort(key=len, reverse=True)
    
    # 3. Use word boundary (\b) to ensure we don't remove parts of actual words
    for f in fillers:
        text = re.sub(r'\b' + re.escape(f) + r'\b', '', text)

    # 4. Collapse multiple spaces and strip again
    text = re.sub(r'\s+', ' ', text).strip()

    return text
