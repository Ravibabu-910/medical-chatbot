class ChatMemory:
    """
    Conversation memory manager
    
    Maintains a sliding window of recent Q&A pairs
    to provide context for follow-up questions
    """
    
    def __init__(self, max_len=5):
        """
        Initialize memory
        
        Args:
            max_len: Maximum number of Q&A pairs to remember
        """
        self.max_len = max_len
        self.history = []

    def add(self, question, answer):
        """Add a Q&A pair to memory"""
        self.history.append(f"User: {question}\nBot: {answer}")
        
        # Remove oldest if exceeds max length
        if len(self.history) > self.max_len:
            self.history.pop(0)

    def get(self):
        """Get formatted conversation history"""
        return "\n".join(self.history)
    
    def clear(self):
        """Clear all memory"""
        self.history.clear()
    
    def size(self):
        """Get number of Q&A pairs in memory"""
        return len(self.history)