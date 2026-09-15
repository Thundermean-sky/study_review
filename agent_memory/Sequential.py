from main import BaseMemoryStrategy


class SequentialMemory(BaseMemoryStrategy):
    def __init__(self):
        self.his = []

    def add_message(self, user_input: str, ai_response: str):
        """
        Adds a new user-AI interaction to the history.
        Each interaction is stored as two dictionary entries in the list.
        """
        self.his.append({
            "role": "user",
            "content": user_input,
        })
        self.his.append({
            "role": "assistant",
            "content": ai_response,
        })

    def get_context(self, query: str) -> str:
        """
        Retrieves the entire conversation history and formats it into a single
        string to be used as context for the LLM. The 'query' parameter is ignored
        as this strategy always returns the full history.
        """
        return "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in self.his])

    def clear(self):
        """Resets the conversation history by clearing the list."""
        self.his = []
        print("Sequential memory cleared")
