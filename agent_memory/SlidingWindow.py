from collections import deque
from main import *

class SlidingWindowMemory(BaseMemoryStrategy):
    def __init__(self, window_size: int = 4):
        """
        Initializes the memory with a deque of a fixed size.

        Args:
            window_size: The number of conversational turns to keep in memory.
                         A single turn consists of one user message and one AI response.
        """
        self.his = deque(maxlen=window_size)

    def add_message(self, user_input: str, ai_response: str):
        self.his.append([
            {
                "role": "user", "content": user_input,
            },
            {
                "role": "assistant", "content": ai_response,
            }
        ])

        print(f"Current store conversation: {self.his}")

    def get_context(self, query: str) -> str:
        """
        Retrieves the conversation history currently within the window and
        formats it into a single string. The 'query' parameter is ignored.
        """
        # Create a temporary list to hold the formatted messages.
        context_list = []
        # Iterate through each turn stored in the deque.
        for turn in self.his:
            # Iterate through the user and assistant messages within that turn.
            for message in turn:
                # Format the message and add it to our list.
                context_list.append(f"{message['role'].capitalize()}: {message['content']}")
        # Join all the formatted messages into a single string, separated by newlines.
        return "\n".join(context_list)

    def clear(self):
        self.his.clear()

