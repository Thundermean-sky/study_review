from main import BaseMemoryStrategy
from SlidingWindow import SlidingWindowMemory
from RetrievalMemory import RetrievalMemory

class HierarchicalMemory(BaseMemoryStrategy):
    def __init__(self, window_size: int = 2, k: int = 3, embedding_dim: int = 1024):
        self.work_memory = SlidingWindowMemory(window_size=window_size)

        self.long_memory = RetrievalMemory(k=k, embedding_dim=embedding_dim)

        self.promoting_keywords = ["remember", "rule", "preference", "always", "never", "allergic"]

    def add_message(self, user_input: str, ai_response: str):
        """
        Adds a message to working memory and conditionally promotes it to long-term
        memory based on its content.
        """
        # All interactions are added to the fast, short-term working memory.
        self.work_memory.add_message(user_input, ai_response)

        # Promotion Logic: Check if the user's input contains a keyword that
        # suggests the information is important and should be stored long-term.
        if any(keyword in user_input.lower() for keyword in self.promoting_keywords):
            print(f"--- [Hierarchical Memory: Promoting message to long-term storage.] ---")
            # If a keyword is found, also add the interaction to the long-term retrieval memory.
            self.long_memory.add_message(user_input, ai_response)

    def get_context(self, query: str) -> str:
        """
               Constructs a rich context by combining relevant information from both
               the long-term and short-term memory layers.
               """
        # Retrieve the most recent conversation from the working memory.
        working_context = self.work_memory.get_context(query)
        # Retrieve semantically relevant facts from the long-term memory based on the current query.
        long_term_context = self.long_memory.get_context(query)

        # Combine both contexts, clearly labeling their sources for the LLM.
        return f"### Retrieved Long-Term Memories:\n{long_term_context}\n\n### Recent Conversation (Working Memory):\n{working_context}"

    def clear(self):
        self.work_memory.clear()
        self.long_memory.clear()